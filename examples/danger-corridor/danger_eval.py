"""Held-out-animal eval for the danger-corridor world -- the pretrain-WIN metric, made causal.

The eval corpus draws from a DISJOINT animal pool (gen_danger.py build ... heldout), so every animal here was
never seen in training. The walk is mechanical; the ONE decision that needs knowledge is the crux tick, where
you stand on the animal's cell and either die (dangerous) or pass (harmless). Danger is never in the text.

We read that decision straight out of the rollout. For each held-out episode:

  roll the model L steps; take the emitted latent at the crux cell k; score it against the row's TWO
  candidates, which share the position prefix "at cell k: the {animal} ..." and differ ONLY in outcome
  (".. attacks, you die" vs ".. is harmless, keep walking"). argmax cosine = the model's predicted outcome.

Because the candidates are position-matched, the accuracy is PURE danger knowledge, not maze skill. If the
model STOPs at or before the crux, that is itself a death prediction (it never reached the animal alive).
Truth = the hidden danger label. Balanced corpus -> chance = 0.5. Reports overall accuracy plus per-class
recall (dangerous: predicts death; safe: predicts pass) and a secondary rollout-length outcome check.

  python danger_eval.py --data danger_eval.npz --ckpt danger_model
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
    p.add_argument("--data", required=True, help="HELD-OUT eval corpus npz from gen_danger.py")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--max-rows", type=int, default=1000)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    z = np.load(a.data, allow_pickle=True)
    seeds = [str(s) for s in list(z["seed"])][:a.max_rows]
    danger = [int(x) for x in list(z["danger"])][:a.max_rows]
    crux = [int(x) for x in list(z["crux_cell"])][:a.max_rows]
    die_txt = [str(x) for x in list(z["crux_die"])][:a.max_rows]
    cont_txt = [str(x) for x in list(z["crux_cont"])][:a.max_rows]
    L = int(z["corridor_len"][0])

    m = LangSetModel.load(a.ckpt, device=a.device)
    m.eval()

    # encode both position-matched crux candidates per row -> the only difference is the danger outcome
    die_emb = torch.cat([F.normalize(m.emit(die_txt[s:s + 256]).float(), dim=-1).cpu()
                         for s in range(0, len(die_txt), 256)])
    cont_emb = torch.cat([F.normalize(m.emit(cont_txt[s:s + 256]).float(), dim=-1).cpu()
                          for s in range(0, len(cont_txt), 256)])

    preds_die, len_pred_die = [], []                             # crux-latent decision; rollout-length decision
    crux_ent, walk_ent = [], []                                  # cat-in-the-box: FSQ entropy at the animal vs a plain step
    B = 64
    for s0 in range(0, len(seeds), B):
        chunk = seeds[s0:s0 + B]
        Lat, lengths, _sL, ent = m.rollout(chunk, max_steps=L, return_lengths=True, return_soft=True)
        Lat = F.normalize(Lat.float(), dim=-1).cpu()             # [b, L, d]
        ent = ent.float().cpu().numpy()                          # [b, L] per-tick FSQ entropy
        for j, gi in enumerate(range(s0, s0 + len(chunk))):
            k, ln = crux[gi], int(lengths[j])
            if ln <= k:                                          # stopped at/before the animal -> a death prediction
                preds_die.append(1)
            else:
                sim_die = float(die_emb[gi] @ Lat[j, k])
                sim_cont = float(cont_emb[gi] @ Lat[j, k])
                preds_die.append(1 if sim_die > sim_cont else 0)
            len_pred_die.append(1 if ln < L else 0)              # secondary: did the walk end before the exit?
            if ln >= 1:                                          # entropy at the crux (the box) vs the first walk step
                crux_ent.append(float(ent[j, min(k, ln - 1)]))
                walk_ent.append(float(ent[j, 0]))
            else:
                crux_ent.append(float("nan")); walk_ent.append(float("nan"))

    y = np.array(danger); pd = np.array(preds_die); pl = np.array(len_pred_die)
    ce = np.array(crux_ent); we = np.array(walk_ent); ok = (pd == y)
    box = {
        "crux_entropy": round(float(np.nanmean(ce)), 4),                 # the cat in the box: how open is die/survive
        "walk_entropy": round(float(np.nanmean(we)), 4),                 # a certain step -> the model's entropy floor
        "excess_at_crux": round(float(np.nanmean(ce - we)), 4),          # lift at the decision (knowledge collapses it)
        "crux_entropy_when_correct": round(float(np.nanmean(ce[ok])), 4) if ok.any() else None,
        "crux_entropy_when_wrong": round(float(np.nanmean(ce[~ok])), 4) if (~ok).any() else None,
    }
    def stats(pred):
        acc = float(np.mean(pred == y))
        dang = float(np.mean(pred[y == 1] == 1)) if (y == 1).any() else float("nan")   # recall on dangerous
        safe = float(np.mean(pred[y == 0] == 0)) if (y == 0).any() else float("nan")   # recall on safe
        return {"acc": round(acc, 4), "dangerous_recall": round(dang, 4), "safe_recall": round(safe, 4)}

    out = {"ckpt": a.ckpt, "n": len(y), "corridor_len": L, "frac_dangerous": round(float(y.mean()), 4),
           "chance": 0.5, "crux_decision": stats(pd), "rollout_length_decision": stats(pl),
           "cat_in_the_box": box}
    print("=== DANGER-CORRIDOR HELD-OUT DIE-VS-SURVIVE ===")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
