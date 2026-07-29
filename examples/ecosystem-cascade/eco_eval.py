"""Held-out-species eval for the trophic-cascade world model -- the pretrain-WIN metric.

The eval corpus draws from a DISJOINT species pool (gen_eco.py build ... heldout), so a species here was
never seen in training. Predicting its survival requires knowing its trophic role, which is never in the text
-- only pretraining supplies it for a novel species. We measure NEXT-STATE RETRIEVAL, broken down by hop:

  roll each held-out seed -> emitted latent per tick; encode every distinct survivor-set text in the corpus
  as a bank; for each tick t>=1, rank the TRUE survivor-set among the bank by cosine to the emitted latent.

  hop 1 (tick 1) needs one step of role reasoning; hop 2 (tick 2) needs tick 1 correct first, so the
  pretrained-vs-random gap should WIDEN with hop. Reports MRR + top-1 per hop.

  python eco_eval.py --data eco_eval.npz --ckpt eco_model
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from langset import LangSetModel


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="HELD-OUT eval corpus npz from gen_eco.py")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--max-rows", type=int, default=1000)
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    z = np.load(a.data, allow_pickle=True)
    seeds = [str(s) for s in list(z["seed"])][:a.max_rows]
    futs = [[str(t) for t in list(ft)] for ft in list(z["fut_text"])][:a.max_rows]

    m = LangSetModel.load(a.ckpt, device=a.device)
    m.eval()

    # bank of every distinct survivor-set description in the corpus -> target latents (the retrieval candidates)
    bank = sorted({t for ft in futs for t in ft})
    idx_of = {t: i for i, t in enumerate(bank)}
    T = []
    for s in range(0, len(bank), 256):
        T.append(F.normalize(m.emit(bank[s:s + 256]).float(), dim=-1).cpu())
    T = torch.cat(T)                                             # [Nbank, d]
    print(f"[eco-eval] {len(seeds)} held-out rollouts | bank={len(bank)} distinct sets | chance top1={1/len(bank):.4f}",
          flush=True)

    # roll the model, score next-state retrieval per hop
    per_hop: dict[int, list[tuple[int, float]]] = {}
    B = 64
    for s0 in range(0, len(seeds), B):
        chunk = seeds[s0:s0 + B]
        L, lengths = m.rollout(chunk, max_steps=a.max_steps, return_lengths=True)
        L = F.normalize(L.float(), dim=-1).cpu()                 # [b, T, d]
        for j, gi in enumerate(range(s0, s0 + len(chunk))):
            ln = int(lengths[j]); ft = futs[gi]
            for t in range(1, min(ln, len(ft))):                 # t=0 is the initial set (echo of input) -> skip
                if ft[t] not in idx_of:
                    continue
                if ft[t] == ft[t - 1] or ft[t].endswith("(none)"):
                    continue                                     # skip trivial ticks: no-change echo, or extinction

                sims = T @ L[j, t]                               # [Nbank]
                order = torch.argsort(sims, descending=True)
                true_i = idx_of[ft[t]]
                rank = int((order == true_i).nonzero()[0]) + 1
                per_hop.setdefault(t, []).append((1 if rank == 1 else 0, 1.0 / rank))

    out = {"ckpt": a.ckpt, "n_rollouts": len(seeds), "bank": len(bank), "chance_top1": round(1 / len(bank), 4),
           "per_hop": {}, "overall": {}}
    all_top1, all_mrr = [], []
    for t in sorted(per_hop):
        arr = per_hop[t]
        top1 = float(np.mean([x[0] for x in arr])); mrr = float(np.mean([x[1] for x in arr]))
        out["per_hop"][t] = {"n": len(arr), "top1": round(top1, 4), "mrr": round(mrr, 4)}
        all_top1 += [x[0] for x in arr]; all_mrr += [x[1] for x in arr]
    out["overall"] = {"n": len(all_top1), "top1": round(float(np.mean(all_top1)), 4),
                      "mrr": round(float(np.mean(all_mrr)), 4)}
    print("=== ECOSYSTEM HELD-OUT NEXT-STATE RETRIEVAL ===")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
