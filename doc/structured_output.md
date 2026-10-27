# Native JSON response formats

The Qwen and GLM chat/completion servers accept `response_format` with type
`json_object` or `json_schema`. LLGuidance constrains token selection after the
reasoning phase, and jsonschema independently validates the completed object.
Install the optional validator alongside the ordinary dependencies:

```bash
pip install -r requirements.txt -r requirements-json.txt
```

Each request owns its matcher. Reasoning uses the original tokenizer, EOS is
blocked until a complete JSON answer, and restart/rewind/requeue preserve the
filter phase. Literal control-tag text inside JSON strings uses ordinary BPE
tokens without changing vocabulary or EOS IDs. Ordinary text requests retain
their existing paths. Missing optional validation dependencies return HTTP 503
for a structured request.

For example, add this field to a normal chat completion request:

```json
{
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "answer",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": false
      }
    }
  }
}
```

`json_object` needs only `{"type": "json_object"}`. A json_schema request
requires a nonempty name, an object-root schema, and an optional boolean strict
field. The accepted keyword set is explicit in `tools/structured_output.py`;
unknown constraints return 400. Schemas are limited to 64 KiB, depth 32 and
length/item bounds of 4096. Only local references are accepted.

This version rejects tools, non-none tool_choice, nonempty stop and ignore_eos
with structured output. JSON SSE buffers content through final validation;
ordinary text/tool SSE remains incremental. Incomplete JSON returns 502, or an
SSE error if headers were already sent, instead of a successful partial answer.

## CPU validation

```bash
QWEN_JSON_MODEL=/path/to/qwen/tokenizer-and-config \
GLM_JSON_MODEL=/path/to/glm/tokenizer-and-config \
  python tests/test_native_json.py -v
```

These are metadata/tokenizer fixtures, not weight directories to load. The
suite exercises actual tokenization and grammar masks, reasoning/EOS,
restart/rewind/requeue, concurrent matcher ownership, literal control tags,
logprobs and fake-engine HTTP/SSE handling. It does not qualify model quality,
GPU kernels or long contexts. No HIP/C++ ABI changes are included.
