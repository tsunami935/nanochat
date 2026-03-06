"""
Distributed dataloaders for pretraining.

BOS-aligned bestfit:
   - Every row starts with BOS token
   - Documents packed using best-fit algorithm to minimize cropping
   - When no document fits remaining space, crops a document to fill exactly
   - 100% utilization (no padding), ~35% tokens cropped at T=2048

Compared to the original tokenizing_distributed_data_loader:
BOS-aligned loses ~35% of tokens to cropping, but ensures that
there are fewer "confusing" tokens in the train/val batches as every token can
now attend back to the BOS token and sees the full context of the document.

Fallback to the original if you have very limited data AND long documents:
https://github.com/karpathy/nanochat/blob/3c3a3d7/nanochat/dataloader.py#L78-L117
"""
import torch
import pyarrow.parquet as pq
from nanochat.common import get_dist_info
from nanochat.tinyshakespeare import load_dataset

def _document_batches(split, epoch=1, tokenizer_batch_size=128):
    """
    Iterator over document batches.

    Handles DDP sharding and approximate resume. Each yield is (text_batch, epoch)
    where text_batch is a list of document strings
    and epoch counts how many times we've cycled through the dataset (starts at 1).
    """
    import random
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    warn_on_legacy = ddp_rank == 0 and split == "train" # rank 0 on train split will warn on legacy

    train_text, val_text = load_dataset()
    text = train_text if split == "train" else val_text
    text = text.split("\n\n")

    first_pass = True
    batches = []

    random.seed(0)
    while True:  # iterate infinitely (multi-epoch)
        if first_pass is True:
            text_idx = 0
            while text_idx < len(text):
                batch = []
                for batch_idx in range(tokenizer_batch_size):
                    if text_idx >= len(text):
                        continue
                    batch.append(text[text_idx])
                    text_idx += 1

                    while random.random() < 0.6:
                        if text_idx >= len(text):
                            break
                        batch[batch_idx] += "\n\n" + text[text_idx]
                        text_idx += 1
                    batch_idx += 1
                yield batch, epoch
                batches.append(batch)
            del train_text
            del val_text
        else:
            for batch in batches:
                yield batch, epoch
        first_pass = False
        epoch += 1

def tokenizing_distributed_data_loader_with_state(tokenizer, B, T, split, tokenizer_threads=4, tokenizer_batch_size=128, device="cuda", epoch=1):
    """
    Stream text, tokenize, yield training batches.

    This is the original dataloader that streams tokens into a flat buffer and reshapes.
    Rows may start mid-document (no guaranteed BOS at position 0).

    Supports approximate resume via state_dict.
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    batches = _document_batches(split, epoch, tokenizer_batch_size)
    needed_tokens = B * T + 1  # +1 for target at last position
    bos_token = tokenizer.get_bos_token_id()
    token_buffer = []
    epoch = 1

    while True:
        # Accumulate enough tokens
        while len(token_buffer) < needed_tokens:
            batch, epoch = next(batches)
            token_lists = tokenizer.encode(batch, prepend=bos_token, num_threads=tokenizer_threads)
            for tokens in token_lists:
                token_buffer.extend(tokens)
        tokens = token_buffer[:needed_tokens] # Read B*T+1 tokens (+1 is only for the target for the last token)
        token_buffer = token_buffer[B*T:] # Advance by B*T tokens, so we move exactly one window of B*T tokens over

        # Package tokens into inputs and targets, yield
        use_cuda = device == "cuda"
        scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda)
        inputs = scratch[:-1].view(B, T).to(device=device, non_blocking=use_cuda)
        targets = scratch[1:].view(B, T).to(device=device, non_blocking=use_cuda)
        yield inputs, targets, epoch


def tokenizing_distributed_data_loader(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, epoch in tokenizing_distributed_data_loader_with_state(*args, **kwargs):
        yield inputs, targets
