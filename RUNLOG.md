# RUNLOG

## Baseline
- Hypothesis: n/a — establish starting point.
- Changed: nothing, ran starter/train.py as given.
- Dev bpb: 2.3718 (params 1,339,840)
- Conclusion: byte-level tokenizer + constant lr/no schedule/no decay/no clip is the floor. Everything questionable in train.py (flagged in handout) is fair game.

## ckpt1 — tokenizer change
- Hypothesis: byte-level tokenizer fragments Devanagari into many single-byte tokens, wasting block_size on partial characters. A trained tokenizer should shorten sequences and raise real content per window.
- Changed: byte tokenizer → BPE tokenizer trained only on train_corpus.txt, vocab_size 512.
- Dev bpb: 2.3718 → 2.2417 (params 1,421,760, eval tokens 159,225 → 79,358 — confirms sequences roughly halved)
- Conclusion: correct call, kept.

## ckpt2 — optimizer/schedule (first attempt)
- Hypothesis: warmup + cosine decay + weight decay + grad clipping should beat constant lr/no decay.
- Changed: Adam → AdamW, added lr_at() cosine schedule, weight_decay, grad_clip. Ran with default args: lr=3e-4, weight_decay=0.1, warmup=100.
- Dev bpb: 2.2417 → 2.3632 (worse)
- Conclusion: regression, not the schedule idea itself. Peak lr unchanged from baseline means cosine decay lowers *average* lr across the run — net less learning in 2000 steps. weight_decay=0.1 is tuned for far longer runs; at 2000 steps it drags weights toward zero faster than gradients can shape them.

## ckpt2b — optimizer/schedule (retuned)
- Hypothesis: raise peak lr and cut weight decay to compensate for the fixed short step budget.
- Changed: lr 3e-4→8e-4, weight_decay 0.1→0.01, warmup 100→50. Same AdamW + cosine + clip code as ckpt2.
- Dev bpb: 2.3632 → 2.1772 (new best)
- Conclusion: kept. Confirms schedule/optimizer change was right in principle — ckpt2's failure was hyperparameter values, not the mechanism.

## ckpt3 — weight tying + init change (combined)
- Hypothesis: tying head/tok_emb frees params for other use; GPT-2-style scaled init trains more stably.
- Changed: tie_weights=True + new init scheme, together, on top of ckpt2b config.
- Dev bpb: 2.1772 → 2.3072 (worse), params 1,339,840
- Conclusion: two interacting changes at once — can't tell which one hurt, or whether they compound. Split into isolated runs.

## ckpt3a — weight tying only
- Changed: tie_weights=True, init unchanged, lr unchanged from ckpt2b.
- Dev bpb: 2.2627 (worse than 2.1772)
- Conclusion: tying alone hurts at this step count. Tied embedding must serve two conflicting roles (input clustering vs output separation) — 2000 steps isn't enough to resolve that tension. Dropped.

## ckpt3b — init change only
- Changed: new init scheme, tying off, lr unchanged from ckpt2b.
- Dev bpb: 2.2647 (worse than 2.1772), params 1,421,760
- Conclusion: scaled/deeper init is a longer-horizon bet — undertrained at 2000 steps vs the flat std=0.05 baseline init. Dropped.

## ckpt4 — block_size increase
- Hypothesis: nothing worth keeping from ckpt3/3a/3b, so re-based on ckpt2b. Larger context should let the model use more history per prediction.
- Changed: block_size 128 → larger, everything else = ckpt2b.
- Dev bpb: 2.1772 → 2.2908 (worse), 163ms/step vs 82ms/step (2x slower)
- Conclusion: bad trade under a fixed step cap — more pos_emb params and compute per step, but corpus/training length too short to actually exploit the extra context. Dropped, reverted to ckpt2b base.

## ckpt5 — RoPE
- Hypothesis: relative positional encoding should generalize better than learned absolute pos_emb.
- Changed: pos_emb → RoPE, everything else = ckpt2b.
- Dev bpb: 2.1772 → 2.3084 (worse)
- Conclusion: RoPE has no learned parameters — the model must learn to exploit relative phase implicitly through Q/K dot products, which takes longer to converge than a directly learnable pos_emb lookup table. RoPE's real advantage (long-range/extrapolation) never gets to pay off in 2000 steps at block_size 128. Dropped, reverted to ckpt2b base.

## ckpt6 — batch size increase
- Hypothesis: steps are capped but batch isn't — a bigger batch increases total tokens seen (steps × batch) within the same step cap, and reduces gradient noise.
- Changed: batch size increased, everything else = ckpt2b.
- Dev bpb: 2.1772 → 1.9687 (new best), 143ms/step (up from 82ms/step)
- Conclusion: single largest win of the whole run. Kept as new base.

## ckpt7 — vocab_size increase (final)
- Hypothesis: raising BPE vocab_size from 512 further shortens sequences (more content per block_size), same reasoning as ckpt1, now stacked on the ckpt6 base.
- Changed: vocab_size 512 → 768, everything else = ckpt6 config (tokenizer, lr/warmup/wd/clip, batch size, no tying, baseline init, learned pos_emb, block_size 128).
- Dev bpb: 1.9687 → 1.9588 (new best, final), params 1,503,680, eval tokens 71,969
- Conclusion: kept. Final submitted checkpoint.

## Ordering note
Architecture/optimizer/init/tying/RoPE/block_size experiments were run first because they're cheap (~75-165ms/step) to test and discard. Batch size and vocab_size were run last and applied only once, on top of the already-best config, because both raise ms/step for every run from that point on — running them early would have taxed the time budget of every subsequent experiment.
