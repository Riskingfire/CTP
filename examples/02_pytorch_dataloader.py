"""Use CTProtocol as a drop-in IterableDataset for a PyTorch DataLoader.

pip install "ctp-training[torch,parquet]" transformers
"""

from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from ctprotocol import CTProtocolConfig
from ctprotocol.integrations.pytorch import CTProtocolIterableDataset

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("gpt2")

dataset = CTProtocolIterableDataset(
    "hf://datasets/org/name/data/train-{00000..00031}-of-00032.parquet",
    columns=["text"],
    text_field="text",
    shuffle_shards=True,
    shuffle_buffer=20_000,
    config=CTProtocolConfig(ahead_seconds=60, max_cache_mb=2048),  # split across workers automatically
)


def collate(texts):
    return tokenizer(texts, truncation=True, max_length=512, padding=True, return_tensors="pt")


loader = DataLoader(dataset, batch_size=32, num_workers=4, collate_fn=collate)

for epoch in range(3):
    dataset.set_epoch(epoch)  # new shard order + shuffle each epoch
    for batch in loader:
        loss = model(**batch, labels=batch["input_ids"]).loss
        loss.backward()
