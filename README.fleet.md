# mikenavi Kyojin fleet source

Upstream1.8.0 with native JSON response_format in committed source. Thinking/EOS,
rollback/requeue, independent final validation and literal control-token handling
are covered by tests/test_native_json.py. No patch application is needed.

`bash local/build.sh --verify-only` checks product bytes. `bash local/build.sh`
installs the hashed JSON dependencies and runs the upstream native build in an
already prepared ROCm/Python3.13 SDK environment. The lock targets Linux x86_64
CPython3.13. The SDK/torch wheel and model artifacts are external dependencies;
no model tensors, credentials, host profiles or caches are stored here.

Set QWEN_JSON_MODEL and GLM_JSON_MODEL to read-only tokenizer/config fixture
roots and run tests/test_native_json.py in the SDK environment without a GPU.
The GLM path is a metadata-only regression fixture, not a fleet GLM deployment.
JSON SSE buffers until validation. JSON with tools, stop or ignore_eos is400;
incomplete JSON is an error. Ordinary text/tool streams retain upstream behavior.

Loop guard: inside a Qwen tool call the answer text/counter rules skip (repeated source paths in a
result file were cut as a loop); an exact period must hold 2048 tokens. tools/qwen/test_guard.py ToolCallTests.

Update by merging/rebasing upstream into this branch, running regressions and
publishing a new immutable fleet tag. Deploy exact commit/image digests. A source
export does not qualify a new compiler, GPU backend or long-context workload.
