"""BPE tokenizer trained only on train_corpus.txt. Byte-level base (256
tokens) + learned merges, so encode/decode is lossless by construction:
decoding always expands ids back to the exact original bytes, regardless
of how pre-tokenization grouped things during training/encoding.
"""
import json
import os
import re
from collections import Counter

BASE_VOCAB = 256
_MERGES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "bpe_merges.json")

_PRETOK_RE = re.compile(r"\S+|\s+")


def _pretokenize(text):
    return _PRETOK_RE.findall(text)


def get_pair_counts(word_freqs):
    counts = Counter()
    for word, freq in word_freqs.items():
        for a, b in zip(word, word[1:]):
            counts[(a, b)] += freq
    return counts


def merge_word(word, pair, new_id):
    out = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


def train_bpe(text, vocab_size, verbose=False):
    assert vocab_size > BASE_VOCAB
    num_merges = vocab_size - BASE_VOCAB

    pretoks = _pretokenize(text)
    word_freqs = Counter()
    for pt in pretoks:
        word_freqs[tuple(pt.encode("utf-8"))] += 1

    merges = []
    next_id = BASE_VOCAB
    for i in range(num_merges):
        pair_counts = get_pair_counts(word_freqs)
        if not pair_counts:
            break
        best_pair = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        new_word_freqs = Counter()
        for word, freq in word_freqs.items():
            new_word_freqs[merge_word(word, best_pair, next_id)] += freq
        word_freqs = new_word_freqs
        merges.append((list(best_pair), next_id))
        if verbose and (i + 1) % 32 == 0:
            print(f"  merge {i+1}/{num_merges}: {best_pair} -> {next_id} "
                  f"(count {pair_counts[best_pair]})")
        next_id += 1
    return merges


class BPETokenizer:
    def __init__(self, merges):
        self.merges = [((a, b), nid) for (a, b), nid in merges]
        self.rank = {pair: i for i, (pair, _nid) in enumerate(self.merges)}
        self.pair_to_id = {pair: nid for pair, nid in self.merges}
        self.vocab_size = BASE_VOCAB + len(self.merges)
        self.id_to_bytes = {i: bytes([i]) for i in range(BASE_VOCAB)}
        for (a, b), nid in self.merges:
            self.id_to_bytes[nid] = self.id_to_bytes[a] + self.id_to_bytes[b]

    def _encode_word(self, word_bytes):
        word = list(word_bytes)
        while len(word) > 1:
            best_rank, best_i = None, None
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                r = self.rank.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_i is None:
                break
            a, b = word[best_i], word[best_i + 1]
            nid = self.pair_to_id[(a, b)]
            word = word[:best_i] + [nid] + word[best_i + 2:]
        return word

    def encode(self, text):
        ids = []
        for pt in _pretokenize(text):
            ids.extend(self._encode_word(tuple(pt.encode("utf-8"))))
        return ids

    def decode(self, ids):
        raw = b"".join(self.id_to_bytes[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    def save(self, path=None):
        path = path or _MERGES_FILE
        with open(path, "w") as f:
            json.dump({"type": "bpe",
                       "merges": [[list(p), nid] for p, nid in self.merges]}, f)


class ByteTokenizer:
    vocab_size = 256

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"type": "byte"}, f)


def load(path=None):
    merges_path = path or _MERGES_FILE
    if os.path.exists(merges_path):
        with open(merges_path) as f:
            data = json.load(f)
        return BPETokenizer(data["merges"])
    return ByteTokenizer()