"""Lossless, bounded spill storage for one Qwen engine's image attachments.

RAM residency does not define an attachment's pseudo token identity. An evicted
MMEmbedding is stored with its exact rows, token string, deepstack and MRoPE
metadata. Only these bytes may restore its IDs; a lost/corrupt entry is a normal
vision miss with fresh IDs. The key includes the occurrence within the prompt.

This is a process-lifetime cache, not a KV snapshot format: an exclusive directory
lease prevents sharing across engines, and startup removes stale spill files.
No model fingerprint is guessed and no old process's token IDs are imported.
The byte quota includes the in-progress write. Small bounded index records hold
paths/checksums, never tensors. Current prompt heads have priority at quota.
"""
from __future__ import annotations

import atexit
import errno
import fcntl
import hashlib
import os
from pathlib import Path
import stat
import uuid

from image_cache import ImageEmbeddingCache


def _positive_env(name, default):
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def make_image_cache():
    """Opt-in spill directory; the ordinary 32-entry / 1 GiB RAM limits stay intact."""
    root = os.environ.get("KYOJIN_ATTACHMENT_CACHE_DIR")
    if not root:
        return ImageEmbeddingCache()
    budget = _positive_env("KYOJIN_ATTACHMENT_CACHE_BYTES", 4 << 30)
    count = _positive_env("KYOJIN_ATTACHMENT_CACHE_ITEMS", 16384)
    try:
        backing = AttachmentStore(Path(root) / "qwen-attachments-v1", budget, count)
    except OSError as exc:
        print(f"qserve: attachment store unavailable ({exc}); using RAM cache", flush=True)
        return ImageEmbeddingCache()
    print(f"qserve: attachment store {backing.root}, cap={budget} bytes/{count} entries; process lifetime", flush=True)
    return ImageEmbeddingCache(backing=backing)


class _QuotaWriter:
    def __init__(self, store, slot, stream):
        self.store, self.slot, self.stream = store, slot, stream

    def write(self, data):
        # torch.save streams forward to file-like objects. Check BEFORE writing,
        # including archive headers/padding, rather than estimating tensor size.
        self.store._reserve(self.slot, self.stream.tell() + len(data))
        return self.stream.write(data)

    def flush(self):
        self.stream.flush()


class AttachmentStore:
    def __init__(self, root, max_bytes=4 << 30, max_items=16384):
        if max_bytes <= 0 or max_items <= 0:
            raise ValueError("attachment store requires positive byte and item limits")
        self.root, self.max_bytes, self.max_items = Path(root), max_bytes, max_items
        self.records, self.order = {}, {}
        self.nbytes, self.serial = 0, 0
        self.stats = dict(hits=0, writes=0, misses=0, errors=0, evictions=0, rejected=0)
        self.epoch = uuid.uuid4().hex
        self._lease = None
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        st = self.root.lstat()
        if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid() or st.st_mode & 0o077:
            raise OSError("attachment directory must be private and owned by the serving user")
        lease = os.open(self.root / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Only files created by this module, after obtaining the lease.
            for entry in self.root.iterdir():
                if self._cache_file(entry):
                    entry.unlink()
        except BaseException:
            os.close(lease)
            raise
        self._lease = lease
        atexit.register(self.close)

    @staticmethod
    def _cache_file(path):
        return path.suffix in (".pt", ".partial") and len(path.stem) == 32 and all(c in "0123456789abcdef" for c in path.stem)

    def set_order(self, order):
        # Keep the first position: repeated images have different occurrence keys.
        self.order = {slot: i for i, slot in reversed(list(enumerate(order)))}

    def _priority(self, slot):
        if slot in self.order:
            return (1, -self.order[slot])
        return (0, self.records.get(slot, {}).get("age", self.serial))

    def _drop(self, slot):
        rec = self.records[slot]
        rec["path"].unlink(missing_ok=True)
        self.nbytes -= rec["size"]
        del self.records[slot]

    def _reserve(self, slot, incoming):
        if incoming > self.max_bytes:
            raise OSError(errno.ENOSPC, "attachment exceeds disk quota")
        while self.nbytes + incoming > self.max_bytes or len(self.records) >= self.max_items:
            victim = min(self.records, key=self._priority, default=None)
            if victim is None or self._priority(victim) > self._priority(slot):
                raise OSError(errno.ENOSPC, "attachment quota protects earlier prompt images")
            self._drop(victim)
            self.stats["evictions"] += 1

    @staticmethod
    def _sha(path):
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()

    def _error(self, exc):
        self.stats["errors"] += 1
        # One visible diagnostic, then counters; a broken disk must not flood logs.
        if self.stats["errors"] == 1:
            print(f"qserve: attachment store error ({exc}); cache misses use fresh IDs", flush=True)

    def store(self, slot, emb):
        if self._lease is None:
            return
        rec = self.records.get(slot)
        if rec and rec["first_index"] == emb.first_index and rec["path"].is_file():
            return
        temporary = self.root / (uuid.uuid4().hex + ".partial")
        try:
            import torch
            if rec:
                self._drop(slot)
            # Clone views so torch.save does not serialize an entire backing
            # storage. No dtype conversion or quantization is allowed here.
            def rows(tensor):
                return tensor.detach().to("cpu").contiguous().clone()
            state = {name: getattr(emb, name) for name in (
                "metadata", "full_length", "mm_length", "first_index", "last_index", "text_alias",
                "grid_thw", "mrope_merge_size", "align", "align_phase", "align_lead")}
            state.update(embeddings=rows(emb.embeddings), token_string=rows(emb.token_string),
                         deepstack_embeddings=([rows(t) for t in emb.deepstack_embeddings]
                                               if emb.deepstack_embeddings is not None else None))
            with temporary.open("xb") as stream:
                os.chmod(temporary, 0o600)
                torch.save(dict(epoch=self.epoch, slot=slot, state=state), _QuotaWriter(self, slot, stream))
            size, checksum = temporary.stat().st_size, self._sha(temporary)
            target = temporary.with_suffix(".pt")
            os.replace(temporary, target)
            self.serial += 1
            self.records[slot] = dict(path=target, size=size, sha256=checksum,
                                      first_index=emb.first_index, age=self.serial)
            self.nbytes += size
            self.stats["writes"] += 1
        except Exception as exc:
            self.stats["rejected"] += 1
            if not isinstance(exc, OSError) or exc.errno != errno.ENOSPC:
                self._error(exc)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as exc:
                self._error(exc)

    def fetch(self, slot):
        rec = self.records.get(slot)
        if rec is None:
            self.stats["misses"] += 1
            return None
        try:
            import torch
            from exllamav3.tokenizer.mm_embedding import MMEmbedding
            path = rec["path"]
            if path.stat().st_size != rec["size"] or self._sha(path) != rec["sha256"]:
                raise ValueError("attachment checksum mismatch")
            data = torch.load(path, map_location="cpu", weights_only=True)
            if data["epoch"] != self.epoch or data["slot"] != slot:
                raise ValueError("attachment belongs to another engine or occurrence")
            state = data["state"]
            emb = MMEmbedding(imp=state)
            emb.token_string = state["token_string"]
            emb.token_list = emb.token_string[0].tolist()
            self.serial += 1
            rec["age"] = self.serial
            self.stats["hits"] += 1
            return emb
        except Exception as exc:
            self._error(exc)
            try:
                self._drop(slot)
            except OSError as drop_exc:
                self._error(drop_exc)
            self.stats["misses"] += 1
            return None

    def clear(self):
        for slot in list(self.records):
            self._drop(slot)
        self.order.clear()

    def close(self):
        if self._lease is not None:
            try:
                self.clear()
            except OSError as exc:
                self._error(exc)
            finally:
                os.close(self._lease)
                self._lease = None
                atexit.unregister(self.close)
