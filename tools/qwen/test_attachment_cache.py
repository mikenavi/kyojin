#!/usr/bin/env python3
"""CPU regression at QwenEngine._embed: real MMEmbedding, fake vision, no GPU."""
import os
import errno
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).parent))
import serve
from exllamav3.tokenizer.mm_embedding import MMEmbedding


class AttachmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="attachments-")
        self.addCleanup(self.tmp.cleanup)
        self.calls = 0

    def engine(self, max_items=32, max_bytes=1 << 30, disk_bytes=1 << 24):
        env = {"KYOJIN_ATTACHMENT_CACHE_DIR": self.tmp.name,
               "KYOJIN_ATTACHMENT_CACHE_BYTES": str(disk_bytes)}
        # The pre-fix server uses its stock cache. This makes the same regression
        # executable on the parent and on the candidate, at the real _embed seam.
        with patch.dict(os.environ, env):
            factory = getattr(serve, "make_image_cache", serve.image_cache.ImageEmbeddingCache)
            cache = factory()
        cache.max_items, cache.max_bytes = max_items, max_bytes
        if getattr(cache, "backing", None):
            self.addCleanup(cache.backing.close)
        eng = object.__new__(serve.QwenEngine)
        eng.emb_cache, eng.vision, eng.tokenizer = cache, self, None
        return eng

    def get_image_embeddings(self, tokenizer, image):
        self.calls += 1
        rows = torch.arange(12, dtype=torch.float32).reshape(3, 4) + self.calls
        emb = MMEmbedding(rows, torch.tensor([[151652, -1, -1, -1, 151653]]),
                          deepstack_embeddings=[rows + 0.125, rows + 0.25],
                          grid_thw=(1, 2, 6), mrope_merge_size=2)
        emb.metadata = {"original_size": (960, 640), "preprocessed_size": (896, 672)}
        return emb

    def test_129_images_preserve_ids_after_ram_eviction(self):
        eng = self.engine()
        images = [(SimpleNamespace(size=(960, 640)), str(i)) for i in range(129)]
        first = eng._embed(images)
        again = eng._embed(images)
        self.assertEqual([e.token_list for e in first], [e.token_list for e in again])
        self.assertEqual(self.calls, 129)
        self.assertLessEqual(len(eng.emb_cache), 32)

    @staticmethod
    def images(*keys):
        return [(SimpleNamespace(size=(960, 640)), key) for key in keys]

    def test_growing_history_180_images(self):
        eng = self.engine(max_items=2)
        previous = []
        for size in (32, 33, 128, 129, 180, 180):
            current = eng._embed(self.images(*map(str, range(size))))
            ids = [e.token_list for e in current]
            self.assertEqual(previous, ids[:len(previous)])
            previous = ids
        self.assertEqual(self.calls, 180)

    def test_lossless_byte_eviction_deepstack_mrope_and_alignment(self):
        eng = self.engine(max_bytes=1)
        first = eng._embed(self.images("a"))[0]
        again = eng._embed(self.images("a"))[0]
        self.assertIsNot(first, again)
        for name in ("full_length", "mm_length", "first_index", "last_index", "text_alias",
                     "metadata", "grid_thw", "mrope_merge_size", "token_list",
                     "align", "align_phase", "align_lead"):
            self.assertEqual(getattr(first, name), getattr(again, name), name)
        for a, b in zip([first.embeddings, first.token_string] + first.deepstack_embeddings,
                        [again.embeddings, again.token_string] + again.deepstack_embeddings):
            self.assertEqual(a.dtype, b.dtype)
            self.assertTrue(torch.equal(a, b))
        self.assertEqual(eng.emb_cache.nbytes, 0)
        self.assertEqual(self.calls, 1)

    def test_repeated_content_has_distinct_stable_occurrences(self):
        eng = self.engine(max_items=1)
        first = eng._embed(self.images("same", "other", "same", "same"))
        again = eng._embed(self.images("same", "other", "same", "same"))
        self.assertEqual([e.token_list for e in first], [e.token_list for e in again])
        self.assertEqual(len({e.first_index for e in first}), 4)
        shorter = eng._embed(self.images("same", "other", "same"))
        self.assertEqual([e.token_list for e in shorter], [e.token_list for e in first[:3]])
        self.assertEqual(self.calls, 4)

    def test_corrupt_or_missing_file_gets_fresh_ids(self):
        eng = self.engine(max_bytes=1)
        for damage in ("corrupt", "missing"):
            first = eng._embed(self.images(damage))[0]
            rec = eng.emb_cache.backing.records[(damage, 0)]
            if damage == "missing":
                rec["path"].unlink()
            else:
                data = bytearray(rec["path"].read_bytes())
                data[len(data) // 2] ^= 1
                rec["path"].write_bytes(data)
            again = eng._embed(self.images(damage))[0]
            self.assertNotEqual(first.first_index, again.first_index)
        self.assertEqual(self.calls, 4)

    def test_disk_quota_keeps_head_and_never_exceeds_bytes(self):
        eng = self.engine(max_bytes=1)
        first = eng._embed(self.images("a"))[0]
        disk = eng.emb_cache.backing
        size = disk.nbytes
        disk.max_bytes = size * 2
        images = self.images("a", "b", "c", "d", "e")
        for _ in range(3):
            result = eng._embed(images)
            self.assertEqual(first.first_index, result[0].first_index)
            self.assertLessEqual(disk.nbytes, disk.max_bytes)
            self.assertEqual(disk.nbytes, sum(f.stat().st_size for f in disk.root.glob("*.pt")))
            self.assertFalse(list(disk.root.glob("*.partial")))
        self.assertGreater(disk.stats["rejected"], 0)

    def test_entry_quota_and_new_conversation(self):
        eng = self.engine(max_bytes=1)
        disk = eng.emb_cache.backing
        disk.max_items = 2
        eng._embed(self.images("a", "b", "c"))
        self.assertEqual(set(disk.records), {("a", 0), ("b", 0)})
        fresh = eng._embed(self.images("x", "y"))
        self.assertEqual(set(disk.records), {("x", 0), ("y", 0)})
        again = eng._embed(self.images("x", "y"))
        self.assertEqual([e.token_list for e in fresh], [e.token_list for e in again])

    def test_failed_write_cannot_keep_identity(self):
        eng = self.engine(max_bytes=1)
        with patch("attachment_cache.os.replace", side_effect=OSError(errno.ENOSPC, "full")):
            first = eng._embed(self.images("a"))[0]
        disk = eng.emb_cache.backing
        self.assertEqual(disk.nbytes, 0)
        self.assertFalse(list(disk.root.glob("*.partial")))
        again = eng._embed(self.images("a"))[0]
        self.assertNotEqual(first.first_index, again.first_index)

    def test_restart_discards_crash_leftovers_and_old_identities(self):
        eng = self.engine(max_bytes=1)
        first = eng._embed(self.images("a"))[0]
        disk = eng.emb_cache.backing
        leftovers = list(disk.root.glob("*.pt"))
        self.assertTrue(leftovers)
        # Simulate process exit without close(), keeping only the spill files.
        os.close(disk._lease)
        disk._lease = None
        restarted = self.engine(max_bytes=1)
        again = restarted._embed(self.images("a"))[0]
        self.assertNotEqual(first.first_index, again.first_index)
        self.assertTrue(all(not f.exists() for f in leftovers))

    def test_other_engine_cannot_reuse_or_delete_active_store(self):
        eng = self.engine(max_bytes=1)
        first = eng._embed(self.images("a"))[0]
        other = self.engine(max_bytes=1)
        self.assertIsNone(other.emb_cache.backing)
        foreign = other._embed(self.images("a"))[0]
        self.assertNotEqual(first.first_index, foreign.first_index)
        again = eng._embed(self.images("a"))[0]
        self.assertEqual(first.first_index, again.first_index)

    def test_threads_share_ids_and_one_vision_pass_per_occurrence(self):
        eng = self.engine(max_bytes=1)
        with ThreadPoolExecutor(max_workers=4) as executor:
            answers = list(executor.map(lambda _: eng._embed(self.images("a", "a")), range(8)))
        self.assertEqual(self.calls, 2)
        for result in answers:
            self.assertEqual([e.token_list for e in result], [e.token_list for e in answers[0]])

    def test_clear_invalidates_disk_identity(self):
        eng = self.engine(max_bytes=1)
        first = eng._embed(self.images("a"))[0]
        eng.emb_cache.clear()
        again = eng._embed(self.images("a"))[0]
        self.assertNotEqual(first.first_index, again.first_index)

    def test_invalid_limits_and_opt_out(self):
        for value in ("0", "-1", "oops", ""):
            with patch.dict(os.environ, {"KYOJIN_ATTACHMENT_CACHE_DIR": self.tmp.name,
                                         "KYOJIN_ATTACHMENT_CACHE_BYTES": value}):
                with self.assertRaises(ValueError):
                    serve.make_image_cache()
        with patch.dict(os.environ, {"KYOJIN_ATTACHMENT_CACHE_DIR": ""}):
            cache = serve.make_image_cache()
        self.assertIsNone(cache.backing)
        self.assertEqual((cache.max_items, cache.max_bytes), (32, 1 << 30))

    def test_wrong_engine_payload_is_a_miss(self):
        eng = self.engine(max_bytes=1)
        first = eng._embed(self.images("a"))[0]
        disk = eng.emb_cache.backing
        rec = disk.records[("a", 0)]
        data = torch.load(rec["path"], weights_only=True)
        data["epoch"] = "another engine"
        torch.save(data, rec["path"])
        rec.update(size=rec["path"].stat().st_size, sha256=disk._sha(rec["path"]))
        disk.nbytes = rec["size"]
        again = eng._embed(self.images("a"))[0]
        self.assertNotEqual(first.first_index, again.first_index)

    def test_alignment_and_dtypes_survive_disk(self):
        eng = self.engine(max_bytes=1)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            def build(*args, **kwargs):
                return MMEmbedding(torch.ones(3, 4, dtype=dtype),
                                   torch.tensor([[151652, 151652, -1, -1, -1, 151653]]),
                                   align=4, align_phase=2, align_lead=1)
            with patch.object(self, "get_image_embeddings", side_effect=build):
                first = eng._embed(self.images(str(dtype)))[0]
                again = eng._embed(self.images(str(dtype)))[0]
            self.assertEqual(first.token_list_at(2), again.token_list_at(2))
            self.assertEqual(again.embeddings.dtype, dtype)
            self.assertTrue(torch.equal(first.embeddings, again.embeddings))

    def test_unavailable_store_falls_back_without_touching_other_files(self):
        root = Path(self.tmp.name) / "qwen-attachments-v1"
        root.write_text("occupied")
        eng = self.engine(max_bytes=1)
        self.assertIsNone(eng.emb_cache.backing)
        first = eng._embed(self.images("a"))[0]
        again = eng._embed(self.images("a"))[0]
        self.assertNotEqual(first.first_index, again.first_index)
        self.assertEqual(root.read_text(), "occupied")

    @unittest.skipUnless(os.environ.get("QWEN_JSON_MODEL"), "real tokenizer fixture required")
    def test_real_prompt_encoding_survives_129_images(self):
        from exllamav3 import Config, Tokenizer
        eng = self.engine(max_items=2)
        eng.tokenizer = Tokenizer(Config.from_directory(os.environ["QWEN_JSON_MODEL"]))
        prompt = "<|im_start|>user\n" + "Look: <|image_pad|>\n" * 129 + "<|im_end|>\n"
        def load(key):
            return self.images(key)[0]
        with patch.object(serve, "load_image_keyed", side_effect=load):
            first, _ = eng._encode(prompt, list(map(str, range(129))))
            again, _ = eng._encode(prompt, list(map(str, range(129))))
        self.assertTrue(torch.equal(first, again))
        self.assertEqual(self.calls, 129)


if __name__ == "__main__":
    unittest.main()
