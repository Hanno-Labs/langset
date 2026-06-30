"""Full GPU validation of langset on the sounds-like data, in ONE A10G job:
  (1) reproduce the specialized geometry (40 epochs) + held-out purity / beats-bootstrap / collapse,
  (2) SetFit composition: build a SetFitModel on the langset body and few-shot classify `acoustic_electronic`,
      vs a raw-MiniLM SetFit baseline (does specializing the geometry help downstream?).

  modal run --detach examples/sounds_like/run_modal.py::go
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
app = modal.App("langset-soundslike")
body_vol = modal.Volume.from_name("langset-soundslike-body", create_if_missing=True)  # persist the trained body

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "transformers>=4.51.0", "peft>=0.13.2", "sentence-transformers>=3.0",
                 "setfit>=1.1.0", "datasets>=2.15.0", "numpy", "scikit-learn", "hf_transfer", "wandb")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(str(REPO / "src/langset"), "/root/langset")
    .add_local_file(str(HERE / "data/prepared.json"), "/root/prepared.json")
)
CARED = ["lead_vocal_gender", "acoustic_electronic", "energy", "tempo", "distortion"]


@app.function(gpu="a10g", image=image, timeout=5400, volumes={"/vol": body_vol})
def run(epochs: int = 40) -> dict[str, Any]:
    import json
    import sys
    import numpy as np
    sys.path.insert(0, "/root")
    from langset import LangSetModel, Trainer, TrainingArguments
    from langset import selection

    def axis_val(rec: dict, ax: str) -> str:
        v = rec.get("axes", {}).get(ax, "unknown")
        return str(v[0]).lower() if isinstance(v, list) and v else str(v).lower()

    recs = json.loads(Path("/root/prepared.json").read_text())
    rows = [{"input_text": r["review"], "target_text": r["fingerprint"],
             **{ax: axis_val(r, ax) for ax in CARED}} for r in recs]

    # (1) reproduce the geometry
    model = LangSetModel.from_pretrained("HuggingFaceTB/SmolLM2-135M",
                                         "sentence-transformers/all-MiniLM-L6-v2", n_emit=4, lora_r=16)
    targs = TrainingArguments(epochs=epochs, batch_size=16, ema=True, ema_m=0.9, lam_anchor=0.1,
                              select="auto", output_dir="/root/out", verbose=True)
    Trainer(model, targs, train_dataset=rows).train()
    model.save_pretrained("/vol/body"); body_vol.commit()    # reuse for other SetFit labels without retraining

    # final held-out geometry eval (review-view vs target-view, in the specialized space)
    rng = np.random.default_rng(0); idx = rng.permutation(len(rows)); val = idx[:96]
    emit_in = model.encode([rows[i]["input_text"] for i in val])
    emit_tg = model.encode([rows[i]["target_text"] for i in val])
    boot = model.bootstrap_encoder.encode([rows[i]["target_text"] for i in val], normalize_embeddings=True)
    labels_val = {ax: [rows[i][ax] for i in val] for ax in CARED}
    geo = selection.evaluate(emit_in, emit_tg, labels_val, boot, "purity")
    print(f"\n[geometry] mrr={geo['mrr']:.3f} collapse={geo['collapse']:.3f} "
          f"purity_mean={geo.get('purity_mean'):.3f} beats_bootstrap={geo.get('beats_bootstrap')}/5", flush=True)

    # (2) SetFit composition: few-shot PRIMARY-GENRE classification on the langset body
    out: dict[str, Any] = {"geometry": {k: geo.get(k) for k in ("mrr", "collapse", "purity_mean", "beats_bootstrap")}}
    try:
        from collections import Counter
        from datasets import Dataset  # type: ignore[import-untyped]
        from setfit import SetFitModel, Trainer as SFT, TrainingArguments as SFA, sample_dataset  # type: ignore[import-untyped]
        prim = lambda r: str(r["genre"]).split(" / ")[0].strip().lower()  # noqa: E731
        counts = Counter(prim(r) for r in recs)
        names = sorted(g for g, c in counts.items() if c >= 15 and g not in ("", "nan"))  # classes w/ enough data
        lab2id = {n: i for i, n in enumerate(names)}
        print(f"[setfit] genre classes ({len(names)}): {names}", flush=True)
        rec2 = [(r["review"], lab2id[prim(r)]) for r in recs if prim(r) in lab2id]
        rng2 = np.random.default_rng(1); perm = rng2.permutation(len(rec2))
        tr, te = perm[:int(0.7 * len(rec2))], perm[int(0.7 * len(rec2)):]
        full_tr = Dataset.from_dict({"text": [rec2[i][0] for i in tr], "label": [rec2[i][1] for i in tr]})
        te_ds = Dataset.from_dict({"text": [rec2[i][0] for i in te], "label": [rec2[i][1] for i in te]})
        few = sample_dataset(full_tr, label_column="label", num_samples=8)
        print(f"[setfit] few-shot train={len(few)} eval={len(te_ds)} ({len(names)} classes)", flush=True)

        def setfit_acc(body: Any, tag: str) -> float:
            clf = SetFitModel(model_body=body, labels=names) if body is not None \
                else SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", labels=names)
            sft = SFT(model=clf, args=SFA(batch_size=16, num_epochs=1, eval_strategy="no"),
                      train_dataset=few, eval_dataset=te_ds, metric="accuracy",
                      column_mapping={"text": "text", "label": "label"})
            sft.train(); acc = float(sft.evaluate()["accuracy"])
            print(f"[setfit] {tag} accuracy = {acc:.3f}", flush=True)
            return acc

        out["setfit_langset_body"] = setfit_acc(model.as_sentence_transformer(), "langset body")
        out["setfit_minilm_baseline"] = setfit_acc(None, "raw MiniLM baseline")
    except Exception as exc:  # noqa: BLE001
        import traceback; traceback.print_exc()
        out["setfit_error"] = str(exc)
    print(f"\n==== RESULT ==== {out}", flush=True)
    return out


@app.local_entrypoint()
def go(epochs: int = 40) -> None:
    print(run.remote(epochs))
