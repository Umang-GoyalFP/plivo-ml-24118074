# NOTES

Final config: BPE tokenizer (vocab_size 768, trained only on train_corpus.txt, lossless byte fallback), AdamW with lr 8e-4, warmup 50, weight_decay 0.01, grad clip 1.0, cosine decay to zero, increased batch size, block_size 128 (unchanged), baseline flat init (std=0.05), no weight tying, learned absolute positional embeddings (no RoPE). Final dev bpb 1.9588 at 1,503,680 params, 2000 steps.

It works because the two biggest levers under a fixed 2000-step cap are: more real content per token (BPE vocab shrinks Devanagari sequences 2-3x) and more tokens per step (bigger batch increases steps × batch = total tokens seen, with less gradient noise). Both directly increase effective training signal within the step budget rather than fighting for it.

Weight tying, scaled init, RoPE, and larger block_size all lost individually — each is a longer-horizon bet that needs more than 2000 steps to pay off, so at this budget the simpler baseline choice (untied, flat init, learned pos_emb, block_size 128) wins by not spending steps recovering from instability.
