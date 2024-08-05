import logging
import time

import numpy as np
import torch
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import TypeVar

def sync_gpus() -> None:
    """Sync all GPUs to make sure all operations are finished, needed for correct benchmarking of latency/throughput."""
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device=i)

T = TypeVar('T')
def map_tensors(obj: T, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> T:
    """Recursively map tensors to device and dtype."""
    if isinstance(obj, torch.Tensor):
        if device is not None:
            obj = obj.to(device=device)
        if dtype is not None:
            obj = obj.to(dtype=dtype)
        return obj
    elif isinstance(obj, (list, tuple)):
        return type(obj)(map_tensors(x, device, dtype) for x in obj)
    elif isinstance(obj, dict):
        return {k: map_tensors(v, device, dtype) for k, v in obj.items()}  # type: ignore
    else:
        return obj

@torch.no_grad()
def evaluate_ppl(
    model: torch.nn.Module, pad_token_id: int | None, testloader: DataLoader[dict[str, torch.Tensor]]
) -> float:
    """
    Evaluate the model's perplexity on the test set using batch processing.
    It is expected that model is already on the correct device.
    """
    sync_gpus()

    start_time = time.time()

    model.eval()

    if pad_token_id:
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_id)
    else:
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    nlls = []

    logging.info("Evaluating perplexity...")
    
    input_ids = testloader.data['input_ids'][0]
    attention_mask = testloader.data['attention_mask'][0]
    tokens_length = 1024
    test_data = []
    for i in range(int(len(input_ids)/tokens_length)):
        test_data_fen = {}
        test_data_fen['input_ids'] = torch.unsqueeze(input_ids[i * tokens_length: (i+1) * tokens_length], dim=0)
        test_data_fen['attention_mask'] = torch.unsqueeze(attention_mask[i * tokens_length: (i+1) * tokens_length], dim=0)
        test_data.append(test_data_fen)

    i = 0
    for batch in test_data:
        i += 1
        logging.debug(f"Evaluating batch {len(nlls)}")
        if model.device.type == 'cuda':
            batch = map_tensors(batch, 'cuda:0')
        # a = model(**batch)
        logits = model(**batch).logits

        # shift outputs and labels autoregressively.
        logits = logits[:, :-1, :]
        shift_labels = batch["input_ids"][:, 1:]

        # CrossEntropyLoss demands data dimension is dimension 1.
        nll = loss_fn(logits.permute(0, 2, 1), shift_labels).float()

        mask = shift_labels != loss_fn.ignore_index
        nll_means = (nll * mask).sum(dim=1) / mask.sum(dim=1)
        nlls.append(nll_means)

    nlls_tensor = torch.cat(nlls)
    ppl = torch.exp(nlls_tensor.mean())

    sync_gpus()

    elapsed = time.time() - start_time
    logging.info(
        "Time spent on evaluation: %s",
        time.strftime("%H:%M:%S.{}".format(str(elapsed % 1)[2:])[:13], time.gmtime(elapsed)),
    )

    return ppl.item()

