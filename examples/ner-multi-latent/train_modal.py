"""Modal runner for the ner-multi-latent demo — full CoNLL-2003 on an A100.

    modal run --detach examples/ner-multi-latent/train_modal.py                       # full run
    modal run --detach examples/ner-multi-latent/train_modal.py --epochs 20 --limit 0

Reports f1 / precision / recall / collapse per epoch to wandb (project: langset-ner-multi-latent).
"""
from __future__ import annotations

from pathlib import Path

import modal

_ROOT = Path(__file__).resolve().parent.parent.parent          # langset repo root

# Standard transformers image. langset's build_backbone loads every backbone via AutoModelForCausalLM + plain PEFT
# LoRA, so no special model-patching wheel is needed. torch is pulled from the cu128 index so the Blackwell (B200)
# GPU is supported; everything else layers on top.
image = (
    modal.Image.debian_slim(python_version="3.11")
    # hf_transfer is ignored by current huggingface_hub -> Xet high-performance is the fast path (law team: 59
    # min/file single-stream -> 91s for all files). Matters for cold-start eval/play containers.
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HOME": "/cache/hf"})
    .pip_install("torch>=2.7.0", extra_index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("transformers>=4.50", "hf_xet", "datasets", "wandb>=0.18", "numpy>=1.26",
                 "peft>=0.13.2", "sentence-transformers>=3.0")
    .add_local_dir(str(_ROOT / "src"), "/pkg/src")
    .add_local_dir(str(_ROOT / "examples" / "ner-multi-latent"), "/pkg/example")
)

app = modal.App("langset-ner-multi-latent")
hf_cache = modal.Volume.from_name("langset-hf-cache", create_if_missing=True)


@app.function(image=image, gpu="B200", timeout=21600,
              volumes={"/cache": hf_cache},
              secrets=[modal.Secret.from_name("wandb-api-key"), modal.Secret.from_name("huggingface")])
def train_ner(limit: int = 0, epochs: int = 15, batch_size: int = 32, max_len: int = 128,
              lr: float = 3e-4, latent_dim: int = 0, fsq_dim: int = 128, fsq_levels: int = 8,
              ema_m: float = 0.99, seed: int = 0,
              out: str = "/cache/ner-best", llm: str = "HuggingFaceTB/SmolLM2-135M") -> None:
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"   # anti-fragmentation for 4B backbones
    import sys
    sys.path.insert(0, "/pkg/src")
    sys.path.insert(0, "/pkg/example")
    from train import build_parser, run  # type: ignore[import-not-found]

    args = build_parser().parse_args([])                        # defaults, then override
    args.llm = llm
    args.limit, args.epochs, args.batch_size, args.max_len, args.lr = limit, epochs, batch_size, max_len, lr
    args.latent_dim = latent_dim                                # 0 => backbone hidden size
    args.fsq_dim, args.fsq_levels, args.ema_m, args.seed = fsq_dim, fsq_levels, ema_m, seed   # token-native FSQ
    args.device = ""                                            # auto -> cuda
    args.wandb = True
    args.out = out                                             # best checkpoint, committed live each improvement
    run(args, commit_fn=hf_cache.commit)


@app.function(image=image, gpu="A100-80GB", timeout=600, volumes={"/cache": hf_cache},
              secrets=[modal.Secret.from_name("huggingface")])
def play(ckpt: str = "/cache/ner-gemma-e2b") -> None:
    """Full entity-bank decode of the current best checkpoint (mirrors the SmolLM play_cpu): roll each sentence
    into entity latents, retrieve each latent's nearest `TYPE: surface` from a bank of gold entities + distractors.
    `modal run ...::play --ckpt /cache/ner-gemma-e2b`."""
    import sys
    sys.path.insert(0, "/pkg/src")
    import torch.nn.functional as F  # noqa: F401
    from langset import LangSetModel  # type: ignore[import-not-found]

    data = [
        ("Barack Obama visited Berlin to meet Angela Merkel .", ["PER: Barack Obama", "LOC: Berlin", "PER: Angela Merkel"]),
        ("Apple and Microsoft announced a partnership in California .", ["ORG: Apple", "ORG: Microsoft", "LOC: California"]),
        ("The United Nations sent aid to Syria and Lebanon .", ["ORG: United Nations", "LOC: Syria", "LOC: Lebanon"]),
        ("Lionel Messi signed with Paris Saint-Germain in France .", ["PER: Lionel Messi", "ORG: Paris Saint-Germain", "LOC: France"]),
        ("Google acquired a London startup for two billion dollars .", ["ORG: Google", "LOC: London"]),
        ("Cristiano Ronaldo scored twice against Manchester United at Old Trafford .",
         ["PER: Cristiano Ronaldo", "ORG: Manchester United", "LOC: Old Trafford"]),
    ]
    distractors = ["PER: Taylor Swift", "ORG: Toyota", "LOC: Tokyo", "MISC: Olympics", "ORG: NASA", "LOC: Amazon River"]

    hf_cache.reload()                                           # pick up the latest committed best
    m = LangSetModel.load(ckpt)
    m.eval()
    dev = m.device
    bank_texts = sorted({e for _, ents in data for e in ents} | set(distractors))
    zb = F.normalize(m.emit(bank_texts).float(), dim=-1).to(dev)   # emit() returns CPU (numpy-compat) -> match rollout's dev
    for s, gold in data:
        lat = F.normalize(m.rollout(s, max_steps=16).float(), dim=-1)    # [L, d] on dev
        sims = lat @ zb.t()
        raw = [bank_texts[int(sims[j].argmax())] for j in range(lat.size(0))]
        hits = len(set(raw) & set(gold))
        print(f"\n{s}\n  gold ({len(gold)}): {gold}\n  decoded : {raw}   ({hits}/{len(gold)})", flush=True)


@app.function(image=image, gpu="A10G", timeout=1800, volumes={"/cache": hf_cache})
def eval_ckpt(ckpt: str = "/cache/ner-best-ss", limit: int = 800) -> None:
    """Diagnose recall: sweep the EOS stop-threshold (higher = emit MORE) and report F1 + the average emitted
    count vs average gold count. If avg-emitted << avg-gold, the model is stopping too early (length problem)."""
    import sys
    sys.path.insert(0, "/pkg/src")
    sys.path.insert(0, "/pkg/example")
    import torch.nn.functional as F  # noqa: F401
    from langset import LangSetModel  # type: ignore[import-not-found]
    import prepare  # type: ignore[import-not-found]

    hf_cache.reload()
    m = LangSetModel.load(ckpt); m.eval()
    val = prepare.rows(limit, "validation")
    bank_texts: list[str] = []; bank_tags: list[tuple[str, str]] = []; seen: set[str] = set()
    gold: list[set[tuple[str, str]]] = []
    for r in val:
        gp: set[tuple[str, str]] = set()
        for (s, t) in r["entities"][:12]:
            gp.add((s.lower(), t)); et = f"{t}: {s}"
            if et not in seen:
                seen.add(et); bank_texts.append(et); bank_tags.append((s.lower(), t))
        gold.append(gp)
    zb = F.normalize(m.emit(bank_texts).float(), dim=-1).to(m.device)
    avg_gold = sum(len(g) for g in gold) / len(gold)

    def score(thresh: float) -> tuple[float, float, float, float]:
        tp = fp = fn = 0; nlat = 0
        for r, gp in zip(val, gold):
            lat = F.normalize(m.rollout(r["input_text"], max_steps=16, stop_threshold=thresh).float(), dim=-1)
            nlat += lat.size(0)
            sims = lat @ zb.t()
            pred = {bank_tags[int(sims[j].argmax())] for j in range(lat.size(0))}
            tp += len(pred & gp); fp += len(pred - gp); fn += len(gp - pred)
        p = tp / max(tp + fp, 1); rr = tp / max(tp + fn, 1)
        return 2 * p * rr / max(p + rr, 1e-9), p, rr, nlat / len(val)

    print(f"avg gold entities/sentence = {avg_gold:.2f}", flush=True)
    for th in (0.0, 1.0, 2.0, 4.0):
        f1, p, rr, avg_len = score(th)
        print(f"stop_thresh={th}: f1={f1:.3f} p={p:.3f} r={rr:.3f}  avg_emitted={avg_len:.2f}", flush=True)


_SPACE_ENTITIES = [
    ("PER", "Lionel Messi"), ("PER", "Cristiano Ronaldo"), ("PER", "Neymar"), ("PER", "Kylian Mbappe"),
    ("PER", "Zinedine Zidane"), ("PER", "Barack Obama"), ("PER", "Angela Merkel"), ("PER", "Vladimir Putin"),
    ("PER", "Joe Biden"), ("PER", "Emmanuel Macron"),
    ("ORG", "Apple"), ("ORG", "Microsoft"), ("ORG", "Google"), ("ORG", "Amazon"), ("ORG", "Samsung"), ("ORG", "IBM"),
    ("ORG", "Manchester United"), ("ORG", "Real Madrid"), ("ORG", "Barcelona"), ("ORG", "Liverpool"), ("ORG", "Arsenal"),
    ("LOC", "France"), ("LOC", "Germany"), ("LOC", "Japan"), ("LOC", "Brazil"), ("LOC", "Syria"), ("LOC", "Lebanon"),
    ("LOC", "Berlin"), ("LOC", "London"), ("LOC", "Tokyo"), ("LOC", "Paris"), ("LOC", "Madrid"),
    ("MISC", "Olympics"), ("MISC", "World Cup"), ("MISC", "Wimbledon"), ("MISC", "Nobel Prize"),
]


@app.function(image=image, gpu="A100-80GB", timeout=900, volumes={"/cache": hf_cache},
              secrets=[modal.Secret.from_name("huggingface")])
def embed_space(ckpt: str) -> dict:
    """Emit the curated entity bank as latents (the space the model retrieves against) and return them for
    local 2D projection. Emits the faithful `TYPE: surface` bank form; within-type geometry is prefix-independent."""
    import sys
    sys.path.insert(0, "/pkg/src")
    import torch.nn.functional as F  # noqa: F401
    from langset import LangSetModel  # type: ignore[import-not-found]

    hf_cache.reload()
    m = LangSetModel.load(ckpt)
    m.eval()
    texts = [f"{t}: {s}" for t, s in _SPACE_ENTITIES]
    z = F.normalize(m.emit(texts).float(), dim=-1)                # [N, d] (cpu)
    return {"entities": [s for _, s in _SPACE_ENTITIES], "types": [t for t, _ in _SPACE_ENTITIES],
            "latents": z.tolist()}


@app.local_entrypoint()
def space(qwen: str = "/cache/ner-qwen-4b", gemma: str = "/cache/ner-gemma-e2b-576",
          out: str = "ner_embed_space.png") -> None:
    """Side-by-side t-SNE of the emitted entity space for two checkpoints, colored by NER type."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    runs = [("Qwen3.5-4B  (f1 climbing)", qwen), ("Gemma E2B  (f1 0.199)", gemma)]
    colors = {"PER": "#e41a1c", "ORG": "#377eb8", "LOC": "#4daf4a", "MISC": "#984ea3"}
    fig, axes = plt.subplots(1, len(runs), figsize=(17, 8))
    for ax, (name, ck) in zip(axes, runs):
        d = embed_space.remote(ck)
        X = np.asarray(d["latents"], dtype=np.float32); types = d["types"]; ents = d["entities"]
        emb = TSNE(n_components=2, perplexity=8, random_state=0, init="pca",
                   learning_rate="auto").fit_transform(X)
        for t, c in colors.items():
            idx = [i for i, tt in enumerate(types) if tt == t]
            ax.scatter(emb[idx, 0], emb[idx, 1], c=c, label=t, s=90, alpha=0.85, edgecolors="white", linewidths=0.5)
        for i, e in enumerate(ents):
            ax.annotate(e, (emb[i, 0], emb[i, 1]), fontsize=7, alpha=0.75,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_title(name, fontsize=13); ax.legend(fontsize=10, loc="best")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("NER emitted-entity latent space (t-SNE, colored by type)", fontsize=15)
    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved {out}", flush=True)


@app.local_entrypoint()
def main(limit: int = 0, epochs: int = 15, batch_size: int = 32, seed: int = 0, latent_dim: int = 0,
         fsq_dim: int = 128, fsq_levels: int = 8, out: str = "/cache/ner-best",
         llm: str = "HuggingFaceTB/SmolLM2-135M") -> None:
    train_ner.remote(limit=limit, epochs=epochs, batch_size=batch_size, seed=seed, latent_dim=latent_dim,
                     fsq_dim=fsq_dim, fsq_levels=fsq_levels, out=out, llm=llm)
