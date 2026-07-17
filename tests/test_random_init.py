"""SMOKE test for the random-init / decoupled-tokenizer control arm: LangSetModel.from_scratch.

`from_scratch` copies an architecture but NOT its weights, drops LoRA (full-parameter training), and lets the
tokenizer be chosen independently of the arch. It is the "does pretraining matter" baseline for the paper's
random-init-vs-pretrained ablation. This fast CPU smoke proves:
  1. VOCAB WIRING  -- the fresh embedding table is sized to the chosen tokenizer and a forward/emit runs finite;
  2. GENUINELY RANDOM  -- a vocab-independent backbone weight differs from the arch's pretrained checkpoint;
  3. NO LORA, FULL TRAIN  -- no adapters, every backbone param requires grad;
  4. PERSISTENCE  -- save/load round-trips the FULL backbone (random weights can't rebuild from an id).

Run:  .venv/bin/python tests/test_random_init.py   (or: uv run python tests/test_random_init.py)
"""

from __future__ import annotations

import os
import tempfile

import torch

from langset import LangSetModel
from langset.modeling import _text_tower

ARCH = os.environ.get("LANGSET_TEST_MODEL", "hf-internal-testing/tiny-random-LlamaForCausalLM")


def _emit(
    model: LangSetModel, text: str = "take worm. you pick up the worm from the ground."
) -> torch.Tensor:
    enc = model.tokenizer(text, return_tensors="pt", padding=True)
    model.eval()
    with torch.no_grad():
        return model(enc["input_ids"].to(model.device), enc["attention_mask"].to(model.device))


def test_from_scratch_vocab_wiring_and_emit() -> None:
    m = LangSetModel.from_scratch(
        ARCH, tokenizer_id=ARCH, device="cpu", arch_overrides={"num_hidden_layers": 2}
    )
    assert m._pretrained is False
    assert m._n_layers == 2, f"arch_override didn't take: {m._n_layers} layers"
    # the fresh embedding table matches the DECOUPLED tokenizer, not a baked vocab
    assert m.embed.weight.shape[0] == len(m.tokenizer), (m.embed.weight.shape[0], len(m.tokenizer))
    assert m.vocab_size == len(m.tokenizer)
    z = _emit(m)
    assert z.shape == (1, m.latent_dim), z.shape
    assert torch.isfinite(z).all(), "random-init emit produced non-finite latent"


def test_from_scratch_is_random_not_loaded() -> None:
    """A vocab-independent backbone weight (q_proj) must differ from the arch's pretrained checkpoint -> the weights
    are random init, not silently loaded."""
    from transformers import AutoModelForCausalLM  # type: ignore[import-untyped]

    m = LangSetModel.from_scratch(ARCH, device="cpu")
    rnd = dict(m.backbone.named_parameters())
    key = next(k for k in rnd if "q_proj.weight" in k)
    pre = dict(_text_tower(AutoModelForCausalLM.from_pretrained(ARCH)).named_parameters())
    assert key in pre, (key, list(pre)[:5])
    assert rnd[key].shape == pre[key].shape
    assert not torch.allclose(rnd[key].detach().cpu(), pre[key].detach().cpu()), (
        "from_scratch q_proj equals the pretrained checkpoint -- weights were loaded, not random"
    )


def test_from_scratch_no_lora_full_train() -> None:
    m = LangSetModel.from_scratch(ARCH, device="cpu")
    names = [n for n, _ in m.backbone.named_parameters()]
    assert not any("lora" in n.lower() for n in names), (
        "random-init backbone must have NO LoRA adapters"
    )
    assert all(p.requires_grad for p in m.backbone.parameters()), (
        "random-init backbone must train fully"
    )


def test_from_scratch_save_load_roundtrip() -> None:
    m = LangSetModel.from_scratch(ARCH, device="cpu", arch_overrides={"num_hidden_layers": 2})
    z0 = _emit(m)
    with tempfile.TemporaryDirectory() as td:
        m.save_pretrained(td)
        # random-init must persist the FULL backbone, not just LoRA
        blob = torch.load(os.path.join(td, "langset.pt"), map_location="cpu", weights_only=False)
        assert "backbone" in blob and "lora" not in blob, (
            "random-init save must carry the full backbone"
        )
        m2 = LangSetModel.load(td, device="cpu")
    assert m2._pretrained is False
    z1 = _emit(m2)
    assert torch.allclose(z0, z1, atol=1e-4), "reloaded random-init model emits a different latent"


def test_from_scratch_trains_through_trainer() -> None:
    """The real experiment path: a random-init model trains end-to-end through the Trainer (EMA-twin default) --
    trainable params move and stay finite. Reuses the golden multi-latent harness."""
    import sys
    import tempfile

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    import test_trainer_multi_characterization as M  # noqa: E402

    from langset import Trainer  # noqa: E402

    m = LangSetModel.from_scratch(
        ARCH, device="cpu", multi_latent=True, arch_overrides={"num_hidden_layers": 2}
    )
    before = torch.cat([p.detach().flatten() for p in m.parameters() if p.requires_grad]).clone()
    with tempfile.TemporaryDirectory() as td:
        Trainer(m, M._args(td), M._dataset()).train()
    after = torch.cat([p.detach().flatten() for p in m.parameters() if p.requires_grad])
    assert torch.isfinite(after).all(), "random-init training produced non-finite params"
    assert float((after - before).abs().sum()) > 0.0, (
        "no trainable params moved -- random-init didn't train"
    )


if __name__ == "__main__":
    for fn in (
        test_from_scratch_vocab_wiring_and_emit,
        test_from_scratch_is_random_not_loaded,
        test_from_scratch_no_lora_full_train,
        test_from_scratch_save_load_roundtrip,
        test_from_scratch_trains_through_trainer,
    ):
        fn()
        print(f"ok: {fn.__name__}")
    print("all random-init smoke tests passed")
