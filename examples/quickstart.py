"""Minimal langset run + drop the result into SetFit as a body.

  python examples/quickstart.py
"""
from langset import LangSetModel, Trainer, TrainingArguments

# rows: input_text -> target_text (+ an optional eval-only geometry label)
data = [
    {"input_text": "Sleep — Dopesmoker", "target_text": "glacial detuned doom-metal riffs, sludgy and hypnotic",
     "mood": "heavy"},
    {"input_text": "Aphex Twin — Selected Ambient Works", "target_text": "warm melodic ambient techno, analog pads",
     "mood": "calm"},
    {"input_text": "Burial — Untrue", "target_text": "crackly nocturnal UK garage, pitched vocal ghosts",
     "mood": "calm"},
    {"input_text": "Electric Wizard — Dopethrone", "target_text": "down-tuned fuzz doom, occult and crushing",
     "mood": "heavy"},
] * 8  # tiny toy set

model = LangSetModel.from_pretrained(
    llm_model="HuggingFaceTB/SmolLM2-135M",                 # any HF causal LM (configurable)
    bootstrap_model="sentence-transformers/all-MiniLM-L6-v2",  # the bootstrap latent model (configurable)
    n_emit=4, lora_r=16, dropout=0.0,
)

args = TrainingArguments(epochs=20, batch_size=8, ema=True, ema_m=0.95, lam_anchor=0.1,
                         select="auto")                     # 'mood' present -> selects on kNN purity
trainer = Trainer(model, args, train_dataset=data)          # 'mood' auto-detected as an eval-only label
trainer.train()

z = model.encode(["Bongripper — Satan Worshipping Doom"])   # input text -> latent (ST-compatible)
print("latent shape:", z.shape)

# --- drop straight into SetFit as the body ---
# from setfit import SetFitModel, Trainer as SFTrainer, TrainingArguments as SFArgs
# body = model.as_sentence_transformer()
# clf = SetFitModel(model_body=body, labels=["heavy", "calm"])
# SFTrainer(clf, ...).train()
