import re
from src.Utilities.GPTDatasetV1 import GPTDatasetV1
from torch.utils.data import DataLoader
import tiktoken

def tokenizer_v1(text: str) -> list:
    preprocesses = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    preprocesses = [item.strip() for item in preprocesses if item.strip()]
    return preprocesses


def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True,
                         drop_last=True, number_of_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=number_of_workers
    )
    return dataloader