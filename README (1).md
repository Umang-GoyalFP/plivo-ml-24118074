# 2,000 Step LLM Speedrun

Small GPT trained from scratch on a mixed English + Hindi corpus, under a hard 2,000 optimizer-step / 2,000,000 parameter cap, CPU only.

**Result:** dev bpb 2.3718 → **1.9588**, params 1,503,680 / 2,000,000, steps 2000 / 2000.

## Structure

```
data/
  train_corpus.txt   mixed English + Hindi text (~7 MB) — the only training data
  dev_eval.txt        held-out text for scoring
starter/
  model.py            GPT (attention, blocks, config)
  tokenizer.py         BPE tokenizer (vocab 768), trained only on train_corpus.txt
  train.py             trainer (AdamW, warmup + cosine decay, grad clip)
  evaluate.py          official scorer, unmodified interface
ckpt.pt                final checkpoint
RUNLOG.md              per-run hypothesis / change / result / conclusion
NOTES.md               final config summary (max 10 sentences)
SUMMARY.html           full write-up of all experiments
```

## Usage

Train:
```
python train.py --data ../data/train_corpus.txt --steps 2000 --out ckpt.pt --lr 8e-4 --weight_decay 0.01 --warmup 50
```

Score:
```
python evaluate.py --checkpoint ckpt.pt --text_file <any_text_file>
```

## What changed from baseline

- Tokenizer: byte-level → BPE (vocab 768), trained only on the corpus, lossless byte fallback
- Optimizer: Adam constant lr → AdamW with warmup + cosine decay + weight decay + grad clipping
- Batch size increased (largest single win, given fixed step cap)
- Weight tying, scaled init, RoPE, larger block_size — tried, reverted (regressed at 2000 steps)

See `RUNLOG.md` for every run's hypothesis/result, and `SUMMARY.html` for the full writeup and reasoning behind the experiment order.

## Constraints

- Max 2,000 optimizer steps, max 2,000,000 parameters
- Training data: `train_corpus.txt` only (tokenizer included)
- Pure PyTorch + numpy + stdlib, CPU only, no pretrained weights
