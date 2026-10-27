"""CPU-only native JSON + actual-tokenizer + fake-engine HTTP qualification.

Run with QWEN_JSON_MODEL / GLM_JSON_MODEL pointing at read-only model directories.
No model weights are loaded and no GPU is required.
"""
from __future__ import annotations
import asyncio
import importlib.util
import json
import os
import re
from pathlib import Path
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools/qwen"))
import structured_output as so
from exllamav3 import Config, Tokenizer
from exllamav3.generator.filter.structured import ReasoningJSONFilter
from exllamav3.generator.filter.filter import Filter
from exllamav3.generator.filter.llguidance import LLGuidanceFilter
from exllamav3.generator.job import Job
import torch

SCHEMA = {"type": "object", "properties": {"ответ": {"type": "string"}},
          "required": ["ответ"], "additionalProperties": False}
FORMAT = {"type": "json_schema", "json_schema": {"name": "answer", "strict": True, "schema": SCHEMA}}
TEXT = '{"ответ":"Привет, ёж! <think> </think> <tool_call>"}'
TEMPLATE = ('{% for m in messages %}{{ m.content }}{% endfor %}'
            '{% if add_generation_prompt %}<think>{% endif %}')


def load_server(name, directory):
    sys.path.insert(0, str(ROOT / "tools" / directory))
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / directory / "serve.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


QWEN = load_server("json_qwen_server", "qwen")
GLM = load_server("json_glm_server", "glm")


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizers = [Tokenizer(Config.from_directory(os.environ[k]))
                          for k in ("QWEN_JSON_MODEL", "GLM_JSON_MODEL")]

    def allowed(self, f, token):
        mask = f.get_next_logit_mask().numpy().view('uint32').reshape(-1)
        return token // 32 < len(mask) and bool(int(mask[token // 32]) & (1 << (token % 32)))

    def feed(self, f, t, text):
        ids = (t.encode(text, encode_special_tokens=False, add_bos=False).flatten().tolist() if f._reasoning else
               f._inner._ll_tokenizer.tokenize_str(text))
        for token in ids:
            self.assertTrue(self.allowed(f, token), token)
            f.feed(token)
        return ids

    def test_both_tokenizers_reasoning_eos_guard_close_json_and_rewind(self):
        for t in self.tokenizers:
            with self.subTest(model=t.config.architecture):
                f = ReasoningJSONFilter(t, SCHEMA, True)
                for eos in t.config.eos_token_id_list:
                    self.assertFalse(self.allowed(f, eos))
                reason = self.feed(f, t, "Думаю.")
                f.feed(f.close_id)
                self.assertFalse(f.is_completed())
                ids = self.feed(f, t, TEXT)
                self.assertTrue(f.is_completed())
                self.assertFalse(f.is_active)
                f.rewind(len(ids) + 1)
                self.assertTrue(f.is_active)
                self.assertTrue(f._reasoning)
                f.feed(f.close_id)
                self.feed(f, t, TEXT)
                self.assertTrue(f.is_completed())
                f.restart(False)
                self.assertFalse(f._reasoning)
                self.feed(f, t, TEXT)
                self.assertTrue(f.is_completed())
                f.restart(True)
                f.rewind(0)
                self.assertEqual(f._journal, [])
                with self.assertRaisesRegex(RuntimeError, "EOS before"):
                    f.feed(t.eos_token_id)

    def test_background_masks_owned_and_concurrent_matchers_independent(self):
        t = self.tokenizers[0]
        a, b = (ReasoningJSONFilter(t, SCHEMA, True) for _ in range(2))
        with ThreadPoolExecutor(2) as pool:
            ma = pool.submit(a.get_next_logit_mask).result()
            mb = pool.submit(b.get_next_logit_mask).result()
        original = ma.clone()
        a.feed(a.close_id)
        self.assertTrue(torch.equal(ma, original))
        self.assertTrue(b._reasoning)
        self.assertNotEqual(ma.data_ptr(), mb.data_ptr())
        a.restart(False)
        self.feed(a, t, TEXT)
        self.assertFalse(b.is_completed())

    def test_job_requeue_retains_filter_and_does_not_reset(self):
        # Exercise real Job.activate requeue branch. Scheduler copies these same
        # mutable filters; a requeue must not restart the grammar at its root.
        from types import SimpleNamespace
        t = self.tokenizers[0]
        f = ReasoningJSONFilter(t, SCHEMA, False)
        self.feed(f, t, '{"ответ":')
        consumed = f._inner._consumed
        job = object.__new__(Job)
        job.generator = SimpleNamespace(model=SimpleNamespace(output_device='cpu'))
        job.is_requeued = True
        job.filters = [f]
        job.activate()
        self.assertEqual(f._inner._consumed, consumed)
        self.feed(f, t, '"да"}')
        self.assertTrue(f.is_completed())

    def test_literal_const_never_selects_control_ids_negative_control(self):
        literal = 'ёж <think> внутри </think> и <tool_call>'
        schema = {'type': 'object', 'properties': {'текст': {'const': literal}},
                  'required': ['текст'], 'additionalProperties': False}
        text = json.dumps({'текст': literal}, ensure_ascii=False, separators=(',', ':'))
        def bit(mask, token):
            words = mask.numpy().view('uint32').reshape(-1)
            return token // 32 < len(words) and bool(int(words[token // 32]) & (1 << (token % 32)))
        for t in self.tokenizers:
            with self.subTest(model=t.config.architecture):
                f = ReasoningJSONFilter(t, schema, False)
                # Actual installed tokenizers API, not guessed control spellings.
                added = t.tokenizer.get_added_tokens_decoder()
                self.assertTrue(added)
                legacy = LLGuidanceFilter(t, json_schema=schema, eos_after_completed=True)
                if t is self.tokenizers[0]:
                    # Exact pretty-printed token segmentation observed in the
                    # failed GPU request. Compact BPE segmentation bypasses it.
                    prefix_ids = [90, 198, 220, 328, 159723, 6389, 763, 328, 42565, 16467, 220]
                    self.assertEqual(t.tokenizer.decode(prefix_ids, skip_special_tokens=False), '{\n  "текст": "ёж ')
                    for token in prefix_ids:
                        self.assertTrue(bit(legacy.get_next_logit_mask(), token))
                        legacy.feed(token)
                    control = t.tokenizer.token_to_id('<think>')
                    self.assertTrue(bit(legacy.get_next_logit_mask(), control))
                    with self.assertRaisesRegex(RuntimeError, 'matcher error'):
                        legacy.feed(control)
                ids = f._inner._ll_tokenizer.tokenize_str(text)
                self.assertEqual(f._inner._ll_tokenizer.vocab_size, legacy._ll_tokenizer.vocab_size)
                self.assertEqual(set(f._inner._ll_tokenizer.eos_tokens), set(t.config.eos_token_id_list))
                normal = f._inner._ll_tokenizer.tokenize_str('<think>')
                self.assertFalse(set(normal) & f._content_blocked_ids)
                self.assertEqual(f._inner._ll_tokenizer.decode_bytes(normal), b'<think>')
                # The real runtime tokenizer has not been mutated.
                self.assertEqual(t.encode('<think>', encode_special_tokens=True, add_bos=False).flatten().tolist(),
                                 [t.tokenizer.token_to_id('<think>')])
                for token in ids:
                    mask = f.get_next_logit_mask()
                    for control in f._content_blocked_ids:
                        self.assertFalse(bit(mask, control), (control, added[control]))
                    for eos in f._eos:
                        self.assertTrue(bit(f._content_allow_mask, eos))
                    self.assertTrue(bit(mask, token), token)
                    f.feed(token)
                self.assertTrue(f.is_completed())
                f.restart(False)
                with self.assertRaisesRegex(RuntimeError, 'control token'):
                    f.accept_token(t.tokenizer.token_to_id('<think>'))
                f.restart(True)
                self.assertTrue(bit(f.get_next_logit_mask(), f.close_id))
                f.feed(f.close_id)
                self.assertFalse(bit(f.get_next_logit_mask(), f.close_id))

    def test_actual_qwen_generate_budget_restart_reinstates_content_filter(self):
        from types import SimpleNamespace
        t = self.tokenizers[0]
        prepared = so.prepare({'response_format': {'type': 'json_object'}}, t, True)
        phases, jobs, drops = [], [], []
        e = object.__new__(QWEN.QwenEngine)
        e.torch, e.tokenizer, e.eos = torch, t, list(t.config.eos_token_id_list)
        e.draft_model, e.spec_on, e.slot_store, e.sessions = None, False, None, 1
        e.generator, e._routes = SimpleNamespace(num_draft_tokens=0), {}
        e._driver = SimpleNamespace(done=lambda: False)
        e._encode = lambda prompt, images: (torch.tensor([[100, 101, 102]]), [])
        e._sampler = lambda sampling: None
        e.last_stats = {}
        class CaptureJob:
            def __init__(self, **kw):
                self.filters, self.serial_number = kw['filters'], len(jobs)
                phases.append(self.filters[0]._reasoning)
                jobs.append(self)
        e.Job = CaptureJob
        async def post(fn, *args):
            return fn(*args)
        async def drop(job):
            drops.append(job.serial_number)
        async def submit(job):
            q = asyncio.Queue()
            raw = 'A' if len(jobs) == 1 else '{"x":"ok"}'
            ids = t.encode(raw, encode_special_tokens=False, add_bos=False).flatten().tolist()
            for token in ids:
                self.assertTrue(self.allowed(job.filters[0], token))
                job.filters[0].feed(token)
            event = {'job': job, 'serial': job.serial_number, 'text': raw, 'token_ids': torch.tensor(ids)}
            if len(jobs) > 1:
                event.update(eos=True, eos_reason='end_filter', new_tokens=len(ids), cached_tokens=0)
            q.put_nowait([event])
            return q
        e._post, e._drop, e._submit = post, drop, submit
        async def run():
            result = ''
            async for part in e.generate('p', max_tokens=100, sampling={'temperature': 0}, stop=[],
                    reasoning_first=True, thinking_budget=1, loop_guard=False, structured=prepared):
                result = result[:len(result)-int(part)] if isinstance(part, QWEN.Retract) else result + part
            return result
        text = asyncio.run(run())
        self.assertEqual(phases, [True, False])
        self.assertTrue(prepared.filter.is_completed())
        self.assertIn('{"x":"ok"}', text)
        self.assertIn(0, drops)
        self.assertEqual(e.last_stats['eos_reason'], 'end_filter')

    def test_empty_mask_fails_instead_of_sampling(self):
        f = ReasoningJSONFilter(self.tokenizers[0], SCHEMA, True)
        f._reasoning_mask.zero_()
        with self.assertRaisesRegex(RuntimeError, "no valid next token"):
            f.get_next_logit_mask()

    def test_request_schema_rejections_and_text_noop(self):
        t = self.tokenizers[0]
        self.assertFalse(so.requested({}))
        self.assertFalse(so.requested({'response_format': {'type': 'text'}}))
        for value in ([], 'json', {'type': 'unsupported'}):
            with self.assertRaises(so.RequestError):
                so.requested({'response_format': value})
        for extra in ({'tools': [{}]}, {'tool_choice': 'required'}, {'tool_choice': 'auto'},
                      {'tool_choice': {'type': 'function', 'function': {'name': 'x'}}},
                      {'stop': ['x']}, {'ignore_eos': True}):
            with self.subTest(extra=extra), self.assertRaises(so.RequestError):
                so.prepare(dict(response_format=FORMAT, **extra), t, False)
        for constraint in ({'unknownConstraint': True}, {'format': 'date-time'}, {'oneOf': []},
                           {'$ref': 'https://example.com/schema'}, {'$dynamicRef': '#/x'},
                           {'maxLength': 4097}):
            schema = {'type': 'object', 'properties': {'x': dict(type='string', **constraint)}}
            fmt = {'type': 'json_schema', 'json_schema': {'name': 'x', 'schema': schema}}
            with self.subTest(constraint=constraint), self.assertRaises(so.RequestError):
                so.prepare({'response_format': fmt}, t, False)
        for schema in ({'type': 'array'}, {'type': 'object', 'required': 17},
                       {'type': 'object', 'properties': {'x': {'$ref': '#/$defs/nope'}}}):
            fmt = {'type': 'json_schema', 'json_schema': {'name': 'x', 'schema': schema}}
            with self.subTest(schema=schema), self.assertRaises(so.RequestError):
                so.prepare({'response_format': fmt}, t, False)

    def test_admitted_schema_constraints_mask_invalid_values(self):
        t = self.tokenizers[0]
        cases = [({'type': 'integer', 'minimum': 2}, '{"x":1}'),
                 ({'type': 'integer', 'maximum': 2}, '{"x":3}'),
                 ({'type': 'string', 'minLength': 2}, '{"x":"a"}'),
                 ({'type': 'string', 'maxLength': 2}, '{"x":"abc"}'),
                 ({'type': 'string', 'enum': ['ok']}, '{"x":"no"}'),
                 ({'type': 'string', 'pattern': '^a+$'}, '{"x":"bbb"}'),
                 ({'type': 'array', 'items': {'type': 'integer'}, 'minItems': 2}, '{"x":[1]}')]
        for constraint, invalid in cases:
            schema = {'type': 'object', 'properties': {'x': constraint}, 'required': ['x'],
                      'additionalProperties': False}
            fmt = {'type': 'json_schema', 'json_schema': {'name': 'constraint', 'schema': schema}}
            with self.subTest(constraint=constraint):
                try:
                    prepared = so.prepare({'response_format': fmt}, t, False)
                except so.RequestError:
                    continue  # Compiler decline is an explicit 400, never silent unenforced success.
                invalid_masked = False
                for token in t.encode(invalid, encode_special_tokens=False, add_bos=False).flatten().tolist():
                    if not self.allowed(prepared.filter, token):
                        invalid_masked = True
                        break
                    prepared.filter.feed(token)
                self.assertTrue(invalid_masked, constraint)

    def test_final_validation_uses_native_completion_and_preserves_bytes(self):
        p = so.prepare({'response_format': FORMAT}, self.tokenizers[0], False)
        self.feed(p.filter, self.tokenizers[0], TEXT)
        msg = p.message(TEXT, False)
        p.validate(msg, {'structured_complete': True, 'eos_reason': 'max_new_tokens'})
        self.assertEqual(msg['content'], TEXT)
        with self.assertRaises(so.OutputError):
            p.validate(msg, {'structured_complete': False})
        for text in ('```json\n' + TEXT + '\n```', TEXT + ' trailing', '[]',
                     '{"ответ":"a","ответ":"b"}', '{"ответ":4}', '{"ответ":NaN}', '{"ответ":1e999}'):
            with self.subTest(text=text), self.assertRaises(so.OutputError):
                p.validate({'content': text}, {'structured_complete': True})

    def test_message_boundary_survives_rewind_and_job_restart(self):
        reasoning = 'The literal </think> is ordinary reasoning text.\n'
        for tokenizer in self.tokenizers:
            with self.subTest(model=tokenizer.config.architecture):
                prepared = so.prepare({'response_format': FORMAT}, tokenizer, True)
                filt = prepared.filter
                ids = filt._inner._ll_tokenizer.tokenize_str(reasoning)
                self.assertNotIn(filt.close_id, ids)
                for token in ids:
                    filt.feed(token)
                filt.feed(filt.close_id)
                answer_ids = self.feed(filt, tokenizer, TEXT)
                raw = reasoning + '</think>' + TEXT
                self.assertEqual(prepared.message(raw, True)['content'], TEXT)
                self.assertEqual(prepared.message(raw, True)['reasoning_content'], reasoning)
                filt.rewind(len(answer_ids))
                self.feed(filt, tokenizer, TEXT)
                self.assertEqual(filt.content_text(), TEXT)
                filt.restart(False)  # the server inserted the close when recovering a job
                self.feed(filt, tokenizer, TEXT)
                message = prepared.message(reasoning + '</think>\n\n' + TEXT, True)
                self.assertEqual(message['content'], '\n\n' + TEXT)
                prepared.validate(message, {'structured_complete': True})
                with self.assertRaises(so.OutputError):
                    prepared.message(reasoning + '</think>' + TEXT + ' unexpected', True)


class FakeEngine:
    ctx = 20000
    supports_vision = False
    spec_on = True

    def __init__(self, tokenizer, output=None, *, complete=True, eos='end_filter', sessions=1, error=False,
                 reasoning='Мысль.'):
        self.tokenizer, self.output = tokenizer, output
        self.complete, self.eos = complete, eos
        self.calls, self.last_stats = [], {}
        self.sessions, self.error = sessions, error
        self.reasoning = reasoning

    def prompt_ids(self, prompt):
        return self.tokenizer.encode(prompt, encode_special_tokens=True, add_bos=False)

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text, encode_special_tokens=True, add_bos=False).flatten())

    def spec_stats(self):
        return {'speculative': True}

    async def generate(self, prompt, **kw):
        self.calls.append(kw)
        structured = kw.get('structured')
        reasoning = self.reasoning if prompt.rstrip().endswith('<think>') else None
        answer = self.output if self.output is not None else TEXT
        raw = (reasoning + '</think>' if reasoning is not None else '') + answer
        if structured is not None and self.complete:
            ids = []
            if reasoning is not None:
                ids = structured.filter._inner._ll_tokenizer.tokenize_str(reasoning)
                assert structured.filter.close_id not in ids
                ids.append(structured.filter.close_id)
            ids += structured.filter._inner._ll_tokenizer.tokenize_str(answer)
            for token in ids:
                structured.filter.feed(token)
        if self.error:
            yield raw[:4]
            raise RuntimeError('synthetic failure')
        self.last_stats = {'new_tokens': 33, 'prompt_tokens': 7, 'cached_tokens': 0,
                           'eos_reason': self.eos, 'token_ids': []}
        if kw.get('stats') is not None:
            kw['stats'].update(self.last_stats)
        for i in range(0, len(raw), 3):
            await asyncio.sleep(0)
            yield raw[i:i+3]


class HTTPTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = Tokenizer(Config.from_directory(os.environ['QWEN_JSON_MODEL']))
        cls.g = Tokenizer(Config.from_directory(os.environ['GLM_JSON_MODEL']))

    async def request(self, server, tokenizer, *, extra=None, output=None, complete=True, eos='end_filter', sessions=1,
                      error=False, reasoning='Мысль.', template=TEMPLATE):
        from aiohttp.test_utils import TestClient, TestServer
        engine = FakeEngine(tokenizer, output, complete=complete, eos=eos, sessions=sessions, error=error,
                            reasoning=reasoning)
        defaults = dict(QWEN.load_defaults('/nonexistent'), reasoning_effort=None)
        app = (server.create_app(engine, 'm', template, defaults=defaults) if server is QWEN else
               server.create_app(engine, 'm', template))
        async with TestClient(TestServer(app)) as client:
            body = {'model': 'm', 'messages': [{'role': 'user', 'content': 'Верни JSON'}], 'max_tokens': 1000,
                    'response_format': FORMAT}
            body.update(extra or {})
            resp = await client.post('/v1/chat/completions', json=body)
            text = await resp.text()
            return resp.status, resp.headers.get('Content-Type', ''), text, engine

    async def test_both_servers_stream_and_nonstream_exact_budget_and_literal_tags(self):
        for server, tokenizer in ((QWEN, self.t), (GLM, self.g)):
            for stream in (False, True):
                for sessions in (1, 2):
                    with self.subTest(server=server.__name__, stream=stream, sessions=sessions):
                        status, ctype, text, engine = await self.request(server, tokenizer, extra={'stream': stream}, sessions=sessions,
                                                                       eos='max_new_tokens')
                        self.assertEqual(status, 200, text)
                        self.assertIsNotNone(engine.calls[0]['structured'])
                        if not stream:
                            body = json.loads(text)
                            self.assertEqual(body['choices'][0]['message']['content'], TEXT)
                            self.assertEqual(body['choices'][0]['finish_reason'], 'stop')
                        else:
                            frames = [json.loads(x[6:]) for x in text.splitlines() if x.startswith('data: {')]
                            self.assertFalse(any('error' in x for x in frames), text)
                            content = ''.join(x['choices'][0]['delta'].get('content', '') for x in frames)
                            self.assertEqual(content, TEXT)
                            self.assertEqual(frames[-1]['choices'][0]['finish_reason'], 'stop')
                            self.assertIn('data: [DONE]', text)

    async def test_literal_reasoning_tags_do_not_split_the_json_answer(self):
        reasoning = 'The literal </think> is ordinary text here. <think> is another example.\n'
        for server, tokenizer in ((QWEN, self.t), (GLM, self.g)):
            for stream in (False, True):
                for sessions in (1, 2):
                    with self.subTest(server=server.__name__, stream=stream, sessions=sessions):
                        status, _, text, _ = await self.request(server, tokenizer, extra={'stream': stream},
                                                               sessions=sessions, reasoning=reasoning)
                        self.assertEqual(status, 200, text)
                        if not stream:
                            message = json.loads(text)['choices'][0]['message']
                            self.assertEqual(message['content'], TEXT)
                            self.assertEqual(message['reasoning_content'], reasoning)
                        else:
                            frames = [json.loads(x[6:]) for x in text.splitlines() if x.startswith('data: {')]
                            self.assertFalse(any('error' in x for x in frames), text)
                            for field, expected in (('content', TEXT), ('reasoning_content', reasoning)):
                                self.assertEqual(''.join(x['choices'][0]['delta'].get(field, '') for x in frames), expected)
                            self.assertEqual(frames[-1]['choices'][0]['finish_reason'], 'stop')

    async def test_json_without_reasoning_preserves_literal_tags(self):
        for server, tokenizer in ((QWEN, self.t), (GLM, self.g)):
            for stream in (False, True):
                with self.subTest(server=server.__name__, stream=stream):
                    status, _, text, _ = await self.request(server, tokenizer, extra={'stream': stream},
                                                           template=TEMPLATE.replace('<think>', ''))
                    self.assertEqual(status, 200, text)
                    if not stream:
                        self.assertEqual(json.loads(text)['choices'][0]['message']['content'], TEXT)
                    else:
                        frames = [json.loads(x[6:]) for x in text.splitlines() if x.startswith('data: {')]
                        self.assertFalse(any('error' in x for x in frames), text)
                        self.assertEqual(''.join(x['choices'][0]['delta'].get('content', '') for x in frames), TEXT)

    async def test_text_mismatching_accepted_tokens_is_an_output_error(self):
        from aiohttp.test_utils import TestClient, TestServer
        class MismatchedEngine(FakeEngine):
            async def generate(self, prompt, **kw):
                async for piece in super().generate(prompt, **kw):
                    yield piece
                yield ' unexpected'
        for server, tokenizer in ((QWEN, self.t), (GLM, self.g)):
            for stream in (False, True):
                with self.subTest(server=server.__name__, stream=stream):
                    engine = MismatchedEngine(tokenizer)
                    app = (server.create_app(engine, 'm', TEMPLATE,
                           defaults=dict(QWEN.load_defaults('/nonexistent'), reasoning_effort=None)) if server is QWEN else
                           server.create_app(engine, 'm', TEMPLATE))
                    async with TestClient(TestServer(app)) as client:
                        response = await client.post('/v1/chat/completions', json={
                            'model': 'm', 'messages': [{'role': 'user', 'content': 'JSON'}],
                            'response_format': FORMAT, 'max_tokens': 1000, 'stream': stream})
                        text = await response.text()
                        if not stream:
                            self.assertEqual(response.status, 502, text)
                        else:
                            frames = [json.loads(x[6:]) for x in text.splitlines() if x.startswith('data: {')]
                            reasons = [x['choices'][0]['finish_reason'] for x in frames if 'choices' in x]
                            self.assertIn('error', reasons)
                            self.assertNotIn('stop', reasons)
                        self.assertIn('does not match', text)

    async def test_incomplete_or_failure_is_never_announced_success(self):
        for server, tokenizer in ((QWEN, self.t), (GLM, self.g)):
            for stream in (False, True):
                with self.subTest(server=server.__name__, stream=stream):
                    status, _, text, _ = await self.request(server, tokenizer, extra={'stream': stream},
                                                            output='{"ответ":', complete=False, eos='max_new_tokens')
                    if stream:
                        self.assertEqual(status, 200)
                        frames = [json.loads(x[6:]) for x in text.splitlines() if x.startswith('data: {')]
                        reasons = [x['choices'][0]['finish_reason'] for x in frames if 'choices' in x]
                        self.assertIn('error', reasons)
                        self.assertNotIn('stop', reasons)
                    else:
                        self.assertEqual(status, 502, text)
                        self.assertIn('incomplete', text)

    async def test_unsupported_or_invalid_schema_is_400_before_sse_and_generation(self):
        for server, tokenizer in ((QWEN, self.t), (GLM, self.g)):
            cases = ({'tools': [{'type': 'function', 'function': {'name': 'x'}}]},
                     {'tool_choice': 'auto'}, {'stop': ['}']}, {'ignore_eos': True},
                     {'response_format': {'type': 'unknown'}},
                     {'response_format': {'type': 'json_schema', 'json_schema': {'name': 'bad', 'schema': {
                         'type': 'object', 'properties': {'x': {'type': 'string', 'format': 'uuid'}}}}}})
            for case in cases:
                with self.subTest(server=server.__name__, case=case):
                    status, ctype, text, engine = await self.request(server, tokenizer, extra=dict(case, stream=True))
                    self.assertEqual(status, 400, text)
                    self.assertTrue(ctype.startswith('application/json'))
                    self.assertFalse(engine.calls)

    async def test_concurrent_requests_have_independent_matchers(self):
        from aiohttp.test_utils import TestClient, TestServer
        for server, tokenizer in ((QWEN, self.t), (GLM, self.g)):
            engine = FakeEngine(tokenizer, sessions=2)
            app = (server.create_app(engine, 'm', TEMPLATE,
                   defaults=dict(QWEN.load_defaults('/nonexistent'), reasoning_effort=None)) if server is QWEN else
                   server.create_app(engine, 'm', TEMPLATE))
            async with TestClient(TestServer(app)) as client:
                body = {'model': 'm', 'messages': [{'role': 'user', 'content': 'JSON'}],
                        'response_format': FORMAT, 'max_tokens': 1000}
                responses = await asyncio.gather(*(client.post('/v1/chat/completions', json=body) for _ in range(2)))
                for response in responses:
                    payload = await response.json()
                    self.assertEqual(response.status, 200, payload)
                    self.assertEqual(payload['choices'][0]['message']['content'], TEXT)
                self.assertEqual(len(engine.calls), 2)
                self.assertIsNot(engine.calls[0]['structured'].filter, engine.calls[1]['structured'].filter)

    async def test_client_disconnect_cancels_buffered_structured_generation(self):
        from aiohttp.test_utils import TestClient, TestServer
        class WaitingEngine(FakeEngine):
            async def generate(self, prompt, **kw):
                self.calls.append(kw)
                try:
                    await kw['cancel'].wait()
                finally:
                    self.cancelled.set()
                if False:
                    yield ''
        for server, tokenizer in ((QWEN, self.t), (GLM, self.g)):
            engine = WaitingEngine(tokenizer)
            engine.cancelled = asyncio.Event()
            app = (server.create_app(engine, 'm', TEMPLATE,
                   defaults=dict(QWEN.load_defaults('/nonexistent'), reasoning_effort=None)) if server is QWEN else
                   server.create_app(engine, 'm', TEMPLATE))
            async with TestClient(TestServer(app)) as client:
                response = await client.post('/v1/chat/completions', json={
                    'model': 'm', 'messages': [{'role': 'user', 'content': 'JSON'}], 'response_format': FORMAT,
                    'stream': True, 'max_tokens': 1000})
                await response.content.readuntil(b'\n\n')
                response.close()
                await asyncio.wait_for(engine.cancelled.wait(), 3)
                self.assertTrue(engine.calls[0]['cancel'].is_set())

    async def test_ordinary_text_is_not_wired_to_filters(self):
        for server, tokenizer in ((QWEN, self.t), (GLM, self.g)):
            status, _, text, engine = await self.request(server, tokenizer,
                extra={'response_format': {'type': 'text'}}, output='Обычный ответ')
            self.assertEqual(status, 200, text)
            self.assertNotIn('structured', engine.calls[0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
