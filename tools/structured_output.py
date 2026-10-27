"""Shared request contract for native JSON decoding (Qwen and GLM).

Imports the native filter only for a structured request. Ordinary requests retain
exactly their existing generation and rendering paths. JSON streams buffer the
answer through validation: no partial/invalid JSON is announced as success.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any


class RequestError(ValueError):
    status = 400


class UnavailableError(RequestError):
    status = 503


class OutputError(RuntimeError):
    pass


# Every constraint here is checked independently by jsonschema as well as by
# llguidance. Unrecognised constraints are errors, never annotations by accident.
_CONSTRAINTS = {"type", "properties", "required", "additionalProperties", "items",
                "minItems", "maxItems", "minLength", "maxLength", "enum", "const",
                "anyOf", "allOf", "$ref", "$defs", "definitions", "pattern",
                "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"}
_ANNOTATIONS = {"title", "description", "default", "examples", "$comment", "deprecated",
                "readOnly", "writeOnly"}
_MAX_SCHEMA_BYTES = 64 * 1024
_MAX_DEPTH = 32
_MAX_BOUND = 4096


def _schema_nodes(node: Any, depth: int = 0):
    if depth > _MAX_DEPTH:
        raise RequestError("response_format schema exceeds maximum depth 32")
    if isinstance(node, bool):
        return
    if not isinstance(node, dict):
        raise RequestError("response_format schema nodes must be objects or booleans")
    for key, value in node.items():
        if key not in _CONSTRAINTS | _ANNOTATIONS:
            raise RequestError(f"response_format schema keyword {key!r} is not supported")
        if key == "$ref" and (not isinstance(value, str) or not value.startswith("#")):
            raise RequestError("response_format supports only local schema references (#...)")
        if key in ("properties", "$defs", "definitions"):
            if not isinstance(value, dict):
                raise RequestError(f"response_format schema {key} must be an object")
            for sub in value.values():
                _schema_nodes(sub, depth + 1)
        elif key in ("items", "additionalProperties"):
            _schema_nodes(value, depth + 1)
        elif key in ("anyOf", "allOf"):
            if not isinstance(value, list):
                raise RequestError(f"response_format schema {key} must be an array")
            for sub in value:
                _schema_nodes(sub, depth + 1)
        elif key in ("minItems", "maxItems", "minLength", "maxLength"):
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= _MAX_BOUND:
                raise RequestError(f"response_format schema {key} must be an integer in 0..{_MAX_BOUND}")


def requested(body: dict) -> bool:
    fmt = body.get("response_format")
    if fmt is None:
        return False
    if not isinstance(fmt, dict):
        raise RequestError("response_format must be an object")
    kind = fmt.get("type")
    if kind not in ("text", "json_object", "json_schema"):
        raise RequestError("response_format.type must be text, json_object or json_schema")
    return kind != "text"


@dataclass
class PreparedFormat:
    validator: Any
    filter: Any

    def message(self, text: str, reasoning_first: bool) -> dict:
        if not reasoning_first:
            return {"role": "assistant", "content": text}
        if not self.filter.is_completed():
            return {"role": "assistant", "content": "", "reasoning_content": text}
        # Printable tags may also occur in ordinary reasoning or JSON tokens.
        # The filter journal identifies the accepted JSON suffix unambiguously.
        content = self.filter.content_text()
        if not content or not text.endswith(content):
            raise OutputError("structured output does not match the accepted JSON tokens")
        prefix = text[:-len(content)]
        reasoning, sep, whitespace = prefix.rpartition("</think>")
        if not sep or whitespace.strip():
            raise OutputError("structured output is missing its reasoning boundary")
        return {"role": "assistant", "content": whitespace + content, "reasoning_content": reasoning}

    def validate(self, message: dict, stats: dict) -> None:
        if not stats.get("structured_complete"):
            raise OutputError("structured output was incomplete; increase max_tokens (no complete JSON answer)")
        if message.get("tool_calls"):
            raise OutputError("structured output unexpectedly contained tool calls")
        text = message.get("content")
        if not isinstance(text, str):
            raise OutputError("structured output did not contain JSON text")

        def pairs(items):
            obj = {}
            for key, value in items:
                if key in obj:
                    raise ValueError(f"duplicate JSON key {key!r}")
                obj[key] = value
            return obj

        def constant(value):
            raise ValueError(f"invalid JSON constant {value}")

        def finite_float(value):
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("non-finite JSON number")
            return number

        try:
            value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant, parse_float=finite_float)
            if not isinstance(value, dict):
                raise ValueError("the root must be a JSON object")
            error = next(self.validator.iter_errors(value), None)
            if error is not None:
                raise ValueError(error.message)
        except Exception as exc:
            raise OutputError(f"structured output validation failed: {exc}") from exc


def prepare(body: dict, tokenizer: Any, reasoning_first: bool) -> PreparedFormat:
    """Called in a bounded executor before queue admission / SSE headers."""
    if body.get("tools") or body.get("tool_choice") not in (None, "none"):
        raise RequestError("JSON response_format with tools or tool_choice is not supported")
    if body.get("stop") or body.get("ignore_eos"):
        raise RequestError("JSON response_format with stop or ignore_eos is not supported")
    fmt = body["response_format"]
    if fmt["type"] == "json_object":
        schema = {"type": "object"}
    else:
        spec = fmt.get("json_schema")
        if not isinstance(spec, dict) or not isinstance(spec.get("schema"), dict):
            raise RequestError("response_format.json_schema needs a schema object")
        if not isinstance(spec.get("name"), str) or not spec["name"]:
            raise RequestError("response_format.json_schema needs a nonempty name")
        if "strict" in spec and not isinstance(spec["strict"], bool):
            raise RequestError("response_format.json_schema.strict must be boolean")
        schema = spec["schema"]
        # A small, explicit v1 contract; object-only unions can be added with tests.
        if schema.get("type") != "object":
            raise RequestError("response_format.json_schema root must have type object")
    try:
        packed = json.dumps(schema, allow_nan=False, ensure_ascii=False)
    except (ValueError, TypeError) as exc:
        raise RequestError(f"invalid response_format schema: {exc}") from exc
    if len(packed.encode("utf-8")) > _MAX_SCHEMA_BYTES:
        raise RequestError("response_format schema exceeds maximum size 64 KiB")
    _schema_nodes(schema)
    try:
        from jsonschema import Draft202012Validator
        from referencing import Registry
        from referencing.exceptions import NoSuchResource
        from exllamav3.generator.filter.structured import ReasoningJSONFilter
    except ImportError as exc:
        raise UnavailableError("native JSON requires installed jsonschema and llguidance dependencies") from exc

    def no_remote(uri):
        raise NoSuchResource(ref=uri)

    try:
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema, registry=Registry(retrieve=no_remote))
        filt = ReasoningJSONFilter(tokenizer, schema, reasoning_first)
    except Exception as exc:
        raise RequestError(f"invalid or unsupported response_format schema: {exc}") from exc
    return PreparedFormat(validator, filt)
