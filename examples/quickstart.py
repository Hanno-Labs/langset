"""Minimal langset run + drop the result into SetFit as a body.

  python examples/quickstart.py
"""
from langset import LangSetModel, Trainer, TrainingArguments

# rows: input_text (what you read at inference) -> target_text (the description that DEFINES the geometry)
data = [
    {"input_text": "Sleep — Dopesmoker", "target_text": "glacial detuned doom-metal riffs, sludgy and hypnotic"},
    {"input_text": "Aphex Twin — Selected Ambient Works", "target_text": "warm melodic ambient techno, analog pads"},
    {"input_text": "Burial — Untrue", "target_text": "crackly nocturnal UK garage, pitched vocal ghosts"},
    {"input_text": "Electric Wizard — Dopethrone", "target_text": "down-tuned fuzz doom, occult and crushing"},
] * 8  # tiny toy set

model = LangSetModel.from_pretrained("HuggingFaceTB/SmolLM2-135M", lora_r=16)   # any HF causal LM
Trainer(model, TrainingArguments(epochs=20, batch_size=8), train_dataset=data).train()

z = model.encode(["Bongripper — Satan Worshipping Doom"])   # input text -> latent (ST-compatible)
print("latent shape:", z.shape)

# --- drop straight into SetFit as the body ---
# from setfit import SetFitModel
# clf = SetFitModel(model_body=model.as_sentence_transformer(), labels=["heavy", "calm"])
# clf.fit(x_train, y_train, num_epochs=1)
