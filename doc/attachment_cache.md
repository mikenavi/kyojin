# Qwen attachment cache

Long conversations can exceed the image embedding cache's 32-entry or 1 GiB
RAM budget. Re-encoding an evicted image assigns new pseudo token IDs, so an
otherwise identical conversation no longer matches its cached KV prefix.

The Qwen server can spill evicted attachments to bounded local disk storage:

```bash
export KYOJIN_ATTACHMENT_CACHE_DIR=/path/to/private/cache
export KYOJIN_ATTACHMENT_CACHE_BYTES=4294967296
export KYOJIN_ATTACHMENT_CACHE_ITEMS=16384
# Start tools/qwen/serve.py with the usual model arguments.
```

The directory setting opts in; without it the existing RAM-only behavior stays.
Byte and entry limits default to 4 GiB and 16384 and must be positive integers.
The server creates a private `qwen-attachments-v1` subdirectory, which must be
owned by the serving user and inaccessible to other users. One engine holds an
exclusive directory lease. An unavailable directory falls back to the RAM cache.
No extra Python dependency is required; the directory lease uses Linux flock.

The spill retains the exact rows, dtype, pseudo token IDs, token string,
deepstack tensors, MRoPE and alignment metadata. Identical images at different
prompt positions retain separate occurrence identities. Disk admission favors
earlier images in the current prompt, after dropping unreferenced records.
Checksummed records are read with `torch.load(weights_only=True)`. Corruption,
missing files and storage failures cause normal vision misses with fresh IDs.
The byte bound covers in-progress writes; failed temporary-file cleanup stops
new spill writes until restart, while existing valid entries remain readable.

This cache lasts for one engine process. Startup removes stale spill records
after acquiring the lease; it does not restore image IDs from persisted KV
slots across restarts. The full request still holds its embedding rows, so the
cache budgets do not cap peak request RAM. Request logs report vision builds,
RAM/disk occupancy and cumulative spill counters.

## Validation

CPU regression (no weights loaded or GPU needed):

```bash
QWEN_JSON_MODEL=/path/to/qwen/tokenizer-and-config \
  python tools/qwen/test_attachment_cache.py -v
python tools/strix_halo/test_image_cache.py
```

The tokenizer path is optional; without it the tokenization integration test
skips. The other tests cover 129 images, growing histories, exact tensor and
metadata restoration, duplicate occurrences, quotas, corruption, competing
owners and injected filesystem failures.

Against explicitly reserved vision servers, first run the baseline, then the
candidate with spill enabled, using separate output directories:

```bash
python tools/qwen/test_attachment_cache_gpu.py --model MODEL_ID --baseline --output BASELINE_RESULTS
python tools/qwen/test_attachment_cache_gpu.py --model MODEL_ID --output CANDIDATE_RESULTS
```

The client checks `/slots`, sends requests and records results; it never starts
or switches a server. Optional API credentials use `KYOJIN_TEST_API_KEY`.
It checks 129-image prefix reuse, forced-cold answer parity, changed first-image
content and repeated occurrences. The request fixtures and responses are kept
under the requested output directory; authorization headers are not recorded.
