"""Train with transformers.Trainer on a CTProtocol-backed streaming dataset.

pip install "ctp-training[hf]" transformers torch
"""

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from ctprotocol import CTProtocolDataset, to_hf_iterable

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("gpt2")

stream = CTProtocolDataset("https://example.com/train-{000..015}.jsonl.gz", shuffle_shards=True)
train = to_hf_iterable(stream).map(
    lambda ex: tokenizer(ex["text"], truncation=True, max_length=512), batched=True, remove_columns=["text"]
)

Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="out", max_steps=10_000, per_device_train_batch_size=8
    ),  # max_steps is required
    train_dataset=train,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
).train()
