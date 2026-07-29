"""Load the trained ner-multi-latent checkpoint on CPU and decode each emitted latent against an entity bank.

This mirrors the training F1 metric (nearest full "TYPE: surface" entity), which is the decode the model was
actually trained for — not a bare type-prefix anchor.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
from langset import LangSetModel  # noqa: E402

CKPT = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "langset-ner-best")

# (sentence, gold entities). The bank is the union of all gold entities + a few distractors — each emitted
# latent retrieves its nearest "TYPE: surface" from the bank, so we can see WHAT it decoded to.
DATA = [
    ("Barack Obama visited Berlin to meet Angela Merkel .", ["PER: Barack Obama", "LOC: Berlin", "PER: Angela Merkel"]),
    ("Apple and Microsoft announced a partnership in California .", ["ORG: Apple", "ORG: Microsoft", "LOC: California"]),
    ("The United Nations sent aid to Syria and Lebanon .", ["ORG: United Nations", "LOC: Syria", "LOC: Lebanon"]),
    ("Lionel Messi signed with Paris Saint-Germain in France .", ["PER: Lionel Messi", "ORG: Paris Saint-Germain", "LOC: France"]),
    ("Google acquired a London startup for two billion dollars .", ["ORG: Google", "LOC: London"]),
    ("Cristiano Ronaldo scored twice against Manchester United at Old Trafford .",
     ["PER: Cristiano Ronaldo", "ORG: Manchester United", "LOC: Old Trafford"]),
]
DISTRACTORS = ["PER: Taylor Swift", "ORG: Toyota", "LOC: Tokyo", "MISC: Olympics", "ORG: NASA", "LOC: Amazon River"]


def main() -> None:
    m = LangSetModel.load(CKPT, device="cpu")
    m.eval()
    bank_texts = sorted({e for _, ents in DATA for e in ents} | set(DISTRACTORS))
    zb = F.normalize(m.emit(bank_texts).float(), dim=-1)                       # [N, d]
    for s, gold in DATA:
        lat = F.normalize(m.rollout(s, max_steps=16).float(), dim=-1)          # [L, d]
        sims = lat @ zb.t()                                                    # [L, N]
        # (a) raw argmax decode (what training used)
        raw = [bank_texts[int(sims[j].argmax())] for j in range(lat.size(0))]
        raw_hits = len({e for e in raw} & set(gold))                          # unique gold covered
        # (b) greedy UNIQUE assignment: each bank entity used at most once, by descending similarity
        pairs = sorted(((float(sims[j, n]), j, n) for j in range(lat.size(0)) for n in range(len(bank_texts))),
                       reverse=True)
        used_j: set[int] = set(); used_n: set[int] = set(); uniq: dict[int, str] = {}
        for _sc, j, n in pairs:
            if j not in used_j and n not in used_n:
                used_j.add(j); used_n.add(n); uniq[j] = bank_texts[n]
        uniq_dec = [uniq.get(j, "?") for j in range(lat.size(0))]
        uniq_hits = len(set(uniq_dec) & set(gold))
        print(f"\n{s}\n  gold ({len(gold)}): {gold}"
              f"\n  raw argmax : {raw}   ({raw_hits}/{len(gold)} unique)"
              f"\n  dedup uniq : {uniq_dec}   ({uniq_hits}/{len(gold)} unique)")


if __name__ == "__main__":
    main()
