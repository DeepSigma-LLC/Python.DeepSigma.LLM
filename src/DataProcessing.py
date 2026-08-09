import re

def tokenizer_v1(text: str) -> list:
    preprocesses = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    preprocesses = [item.strip() for item in preprocesses if item.strip()]
    return preprocesses