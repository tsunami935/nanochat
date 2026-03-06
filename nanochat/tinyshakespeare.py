import os
import requests
from nanochat.tokenizer import RustBPETokenizer

def load_dataset():
    """Returns [train, val] data"""
    # -----------------------------------------------------------------------------
    # download the tiny shakespeare dataset
    input_file_path = os.path.join('data', 'input.txt')
    if not os.path.exists(input_file_path):
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(input_file_path, 'w', encoding='utf-8') as f:
            f.write(requests.get(data_url).text)

    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]
    return train_data, val_data

def load_tokenizer(proj: str):
    base_dir = get_proj_dir(proj)
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    return RustBPETokenizer.from_directory(tokenizer_dir)

def get_proj_dir(proj: str):
    dir = os.getenv(proj)
    if dir is None:
        dir = proj
    return dir