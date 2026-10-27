# mikenavi Kyojin fleet source

Upstream 1.8.0 with native JSON response_format in committed source. Thinking/EOS,
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

The reviewed source retains pending tool-call openers across long whitespace
and splits structured answers using the filter's accepted tokens. Literal
`</think>` text in reasoning or JSON does not move the reasoning boundary.

The ROCm fp16-prefill hint is backported from upstream commit
`f5e240c9b990dd8282d7a302c063c90f2bd20a54`. It sets `waves_per_eu=2`
for head dimensions up to 256; `EXL3_PF_WAVES_PER_EU=0` restores the compiler
default. This is a selected backport, not the full 1.8.1 release. The GPU
bit-identity test includes the fleet Qwen geometry (24 query heads, 2 KV heads,
head dimension 256). A source export or CPU pass does not qualify this JIT
kernel change on a GPU; the integration candidate needs a separate GPU check.

Update by merging/rebasing upstream into this branch, running regressions and
publishing a new immutable fleet tag. Deploy exact commit/image digests. A source
export does not qualify a new compiler, GPU backend or long-context workload.

Qwen attachment spill: `KYOJIN_ATTACHMENT_CACHE_DIR` opts into exact, bounded
disk storage independent of the stock 32-entry / 1 GiB RAM embedding cache.
`KYOJIN_ATTACHMENT_CACHE_BYTES` (default 4 GiB) and
`KYOJIN_ATTACHMENT_CACHE_ITEMS` (default 16384) must be positive. The store
retains token IDs, rows, deepstack, MRoPE and occurrence identity across RAM
eviction. Corrupt/missing/unadmitted entries get fresh IDs and normal prefill.
A private exclusive directory lease scopes identities to one engine lifetime;
stale spill files are removed on startup, and concurrent owners fall back to RAM.
This does not restore multimodal identities from persisted KV slots after restart.
The directory contains derived attachment data, not a conversation archive.
The full request still holds its embedding rows; cache limits are not peak request
RAM limits. Per-request vision builds and cumulative disk counters are logged.
If a failed write cannot remove its temporary file, further spill writes stop
until restart so the leftover reservation cannot exceed the byte quota. Existing
verified entries remain readable, and misses continue with fresh IDs.
Run `tools/qwen/test_attachment_cache.py` without GPU, setting QWEN_JSON_MODEL
to a tokenizer/config fixture for the real tokenization gate.

For an explicitly reserved idle GPU server, run
`python tools/qwen/test_attachment_cache_gpu.py --model MODEL_ID --output RECEIPTS`.
It sends real vision requests; it does not load or switch the server. Use
`--baseline` against the parent and a different output directory against the
candidate. Optional credentials come from `KYOJIN_TEST_API_KEY`; receipts contain
test requests and replies, never the authorization header. The parent must show
cache loss; the candidate must reuse the prefix, match forced-cold answers,
recognize changed image content, and preserve repeated occurrences.

On 2026-10-27, the Qwen3.8 Flash Next EXL3 / gfx1151 / MTP build passed all seven
candidate vision checks and eleven short JSON/text/tool regressions. In one
129-image comparison, the parent's repeat reused 0 of 9893 prompt tokens
(8.750 s); the candidate reused 9728 tokens (0.570 s), with zero vision builds
on repeat. This is a targeted cache regression, not a workload speed guarantee
or a new 512K quality qualification. CPU checks passed 176 tests with one upstream
template fixture skip. The exact native SDK/ELF and JSON dependencies were retained.

Review fault injection reproduced two storage-failure bugs before the fixes:
an existing-file stat could abort RAM eviction, and failed temporary cleanup
could bypass the disk quota. Both are covered by regressions. The final reviewed
payload passed all 19 attachment tests and repeated the seven GPU cache checks
plus eleven short regressions. Its 129-image repeat reused 9728 of 9893 tokens
(0.576 s wall); no vision re-encoding was needed.
