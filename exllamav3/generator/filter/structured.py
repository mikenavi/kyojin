"""An always-active reasoning/EOS guard followed by native JSON sampling."""
from __future__ import annotations

import json
import threading
from functools import lru_cache
import numpy as np
import torch
from .filter import Filter
from .llguidance import LLGuidanceFilter, create_ll_tokenizer
from llguidance import LLTokenizer, TokenizerWrapper
from tokenizers import Tokenizer as HFTokenizer


class _JSONBPEEncoder:
    """Preserve every original ID/byte, but tokenize forced JSON as plain BPE.

    Removing added tokens from a serialized LLTokenizer directly renumbers the
    remaining added EOS IDs. The supported TokenizerWrapper API keeps the full
    vocabulary separate from the ordinary-text encoding callback instead.
    """
    def __init__(self, tokenizer, original):
        config = json.loads(tokenizer.tokenizer.to_str())
        config["added_tokens"] = []
        self._hf = HFTokenizer.from_str(json.dumps(config))
        self._hf.no_padding()
        self._hf.no_truncation()
        self.eos_token_id = tokenizer.eos_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.special_token_ids = list(tokenizer.tokenizer.get_added_tokens_decoder())
        self.tokens = [original.decode_bytes([i]) for i in range(original.vocab_size)]

    def __call__(self, text):
        return self._hf.encode(text, add_special_tokens=False).ids


@lru_cache(10)
def _content_tokenizer(tokenizer):
    original = create_ll_tokenizer(tokenizer)
    encoder = _JSONBPEEncoder(tokenizer, original)
    eos = list(getattr(tokenizer.config, "eos_token_id_list", None) or [tokenizer.eos_token_id])
    return LLTokenizer(TokenizerWrapper(encoder), eos_token=eos)


class ReasoningJSONFilter(Filter):
    def __init__(self, tokenizer, schema: dict, reasoning_first: bool):
        super().__init__(tokenizer, None, None, True)
        close = tokenizer.encode("</think>", encode_special_tokens=True, add_bos=False).flatten().tolist()
        if len(close) != 1:
            raise ValueError("native JSON requires a single </think> token")
        self.close_id = close[0]
        self._lock = threading.RLock()
        self._inner = LLGuidanceFilter(tokenizer, json_schema=schema, eos_after_completed=True,
                                       ll_tokenizer=_content_tokenizer(tokenizer))
        # LLTokenizer knows only actual defined IDs, unlike padded model output rows.
        vocab = self._inner._ll_tokenizer.vocab_size
        bits = np.zeros(((vocab + 31) // 32,), dtype=np.uint32)
        bits[:] = np.uint32(0xFFFFFFFF)
        if vocab % 32:
            bits[-1] = np.uint32((1 << (vocab % 32)) - 1)
        eos = list(getattr(tokenizer.config, "eos_token_id_list", None) or [tokenizer.eos_token_id])
        for token in eos:
            if token is None or not 0 <= token < vocab:
                raise ValueError("native JSON EOS outside tokenizer vocabulary")
            bits[token // 32] &= ~np.uint32(1 << (token % 32))
        self._eos = set(eos)
        # Added tokens are control IDs, not their printable spelling's UTF-8
        # bytes in llguidance. Its forced-byte fast path can nevertheless put
        # them in the allowed mask (e.g. literal "<think>" inside a JSON const).
        # Require ordinary BPE pieces for that spelling, as Strixite does.
        added = tokenizer.tokenizer.get_added_tokens_decoder()
        self._content_blocked_ids = frozenset(int(t) for t in added if int(t) not in self._eos)
        content_bits = np.full_like(bits, np.uint32(0xFFFFFFFF))
        for token in self._content_blocked_ids:
            if 0 <= token < vocab:
                content_bits[token // 32] &= ~np.uint32(1 << (token % 32))
        self._content_allow_mask = torch.from_numpy(content_bits.view(np.int32)).unsqueeze(0)
        self._reasoning_mask = torch.from_numpy(bits.view(np.int32)).unsqueeze(0)
        self._initial_reasoning = reasoning_first
        self.reset()

    def restart(self, reasoning_first: bool):
        """A new physical Job after loop-guard recovery; not scheduler requeue.

        The caller has dropped/finished the old Job first. Locking also prevents
        a late background mask worker from racing with matcher reset.
        """
        with self._lock:
            self._initial_reasoning = reasoning_first
            self._journal.clear()
            self.is_active = True
            self.reset()

    def reset(self):
        with self._lock:
            self._reasoning = self._initial_reasoning
            self._inner.reset()

    def accept_token(self, token: int):
        with self._lock:
            if self._reasoning:
                if token in self._eos:
                    raise RuntimeError("EOS before a structured answer")
                if token == self.close_id:
                    self._reasoning = False
                return
            if token in self._content_blocked_ids:
                raise RuntimeError("control token in native JSON content")
            self._inner.accept_token(token)

    def get_next_logit_mask(self):
        with self._lock:
            mask = (self._reasoning_mask if self._reasoning else self._inner.get_next_logit_mask()).clone()
            if not self._reasoning:
                mask &= self._content_allow_mask
            if not np.any(mask.numpy()):
                raise RuntimeError("native JSON has no valid next token")
            # Each future owns its result; a restart can never overwrite a
            # returned mask while its cancelled Job is being cleaned up.
            return mask

    def is_completed(self):
        with self._lock:
            return not self._reasoning and self._inner.is_completed()

    # Base Filter.rollback_tokens=False deliberately replays its journal. This
    # restores both phase and matcher correctly when rewinding across </think>.
