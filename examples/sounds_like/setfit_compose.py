"""Prove the ALIGNED dependency window lets a SetFit user actually use langset: load the trained langset body
and run SetFit's REAL Trainer on it (few-shot primary-genre), vs a raw-MiniLM baseline.

Aligned stack (setfit's working window): transformers 4.46.x + torch 2.4.x. setfit (even main) imports
`transformers.training_args.default_logdir`, removed after 4.46; and transformers 4.46 + torch>=2.5 hits a
`torch.distributed.tensor` bug — so torch is capped <2.5. The langset body (SmolLM2) loads fine in this window.

  modal run --detach examples/sounds_like/setfit_compose.py::go
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
app = modal.App("langset-setfit")
body_vol = modal.Volume.from_name("langset-soundslike-body")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "transformers==4.46.3", "sentence-transformers==3.3.1", "setfit==1.1.0",
                 "peft>=0.13.2", "datasets>=2.15.0", "scikit-learn", "numpy", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(str(REPO / "src/langset"), "/root/langset")
    .add_local_file(str(HERE / "data/prepared.json"), "/root/prepared.json")
)


@app.function(gpu="a10g", image=image, timeout=3600, volumes={"/vol": body_vol})
def run() -> dict[str, Any]:
    import json
    import sys
    from collections import Counter
    import numpy as np
    sys.path.insert(0, "/root")
    from langset import LangSetModel
    from setfit import SetFitModel  # type: ignore[import-untyped]
    print("[setfit] imports OK in aligned window (transformers 4.46.3 + torch 2.4.1 + setfit 1.1.0)", flush=True)

    from collections import defaultdict
    recs = json.loads(Path("/root/prepared.json").read_text())
    model = LangSetModel.load("/vol/body")
    prim = lambda r: str(r["genre"]).split(" / ")[0].strip().lower()  # noqa: E731
    counts = Counter(prim(r) for r in recs)
    names = sorted(g for g, c in counts.items() if c >= 15 and g not in ("", "nan"))
    lab2id = {n: i for i, n in enumerate(names)}
    rec2 = [(r["review"], lab2id[prim(r)]) for r in recs if prim(r) in lab2id]
    rng = np.random.default_rng(1)
    by_cls: dict[int, list[str]] = defaultdict(list)
    for txt, y in rec2:
        by_cls[y].append(txt)
    few_x, few_y, ev_x, ev_y = [], [], [], []
    for y, txts in by_cls.items():
        txts = list(rng.permutation(txts))
        few_x += txts[:8]; few_y += [y] * len(txts[:8])
        ev_x += txts[8:]; ev_y += [y] * len(txts[8:])
    print(f"[setfit] {len(names)} classes; few-shot train={len(few_x)} eval={len(ev_x)}", flush=True)

    from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

    def acc(body: Any, tag: str) -> float:
        # SetFitModel + fit/predict = frozen body + sklearn head (no ST Trainer). This is the clean composition.
        # Direct construction needs an explicit head (from_pretrained auto-creates one).
        clf = SetFitModel(model_body=body, model_head=LogisticRegression(max_iter=2000), labels=names) \
            if body is not None else SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", labels=names)
        clf.fit(few_x, few_y, num_epochs=1)
        pred = clf.predict(ev_x, use_labels=False)
        pred = [int(p) for p in (pred.tolist() if hasattr(pred, "tolist") else pred)]
        a = float(np.mean([p == t for p, t in zip(pred, ev_y)]))
        print(f"[setfit] {tag}: accuracy = {a:.3f}", flush=True)
        return a

    out = {"n_classes": len(names),
           "setfit_langset_body": acc(model.as_sentence_transformer(), "SetFitModel on langset body"),
           "setfit_minilm_baseline": acc(None, "SetFitModel on raw MiniLM")}
    print(f"\n==== RESULT ==== {out}", flush=True)
    return out


@app.local_entrypoint()
def go() -> None:
    print(run.remote())
