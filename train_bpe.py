"""One-off: train BPE merges on train_corpus.txt only, save bpe_merges.json
next to tokenizer.py.

    python train_bpe.py --data ../data/train_corpus.txt --vocab_size 512
"""
import argparse
import time

import tokenizer as tokenizer_mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--vocab_size", type=int, default=512)
    args = ap.parse_args()

    text = open(args.data, encoding="utf-8").read()
    t0 = time.time()
    merges = tokenizer_mod.train_bpe(text, args.vocab_size, verbose=True)
    print(f"trained {len(merges)} merges in {time.time()-t0:.1f}s")

    tok = tokenizer_mod.BPETokenizer(merges)
    tok.save()
    print(f"saved -> {tokenizer_mod._MERGES_FILE}")

    sample = text[:200000]
    ids = tok.encode(sample)
    assert tok.decode(ids) == sample, "round-trip FAILED"
    print(f"round-trip OK. vocab_size={tok.vocab_size}  "
          f"sample: {len(sample.encode('utf-8'))} bytes -> {len(ids)} tokens "
          f"({len(ids)/len(sample.encode('utf-8')):.3f} tokens/byte)")


if __name__ == "__main__":
    main()