import importlib
from typing import Any

import torch
from tqdm import tqdm


def train_one_epoch(
        model,
        train_dataloader,
        threshold,
        score,
        criterion,
        optimizer,
        device,
):
    model = model.train()

    train_epoch_loss = torch.as_tensor(0., device=device)

    for batch_idx, (batch_inputs, batch_labels) in enumerate(tqdm(train_dataloader)):
        batch_inputs = batch_inputs.to(device, non_blocking=True)
        batch_labels = batch_labels.float().to(device, non_blocking=True)

        optimizer.zero_grad()

        batch_outputs = model(batch_inputs)
        batch_loss = criterion(batch_outputs, batch_labels)

        batch_loss.backward()
        optimizer.step()

        train_epoch_loss += batch_loss.detach() / len(train_dataloader)

        batch_thresh_outputs = batch_outputs > threshold
        batch_score_train = score(batch_labels, batch_thresh_outputs)
    score_epoch_train = score.compute()

    return score_epoch_train, train_epoch_loss


def validate_one_epoch(
        model,
        val_dataloader,
        threshold,
        score,
        criterion,
        device,
):
    model = model.eval()

    val_epoch_loss = torch.as_tensor(0., device=device)

    for batch_idx, (batch_inputs, batch_labels) in enumerate(tqdm(val_dataloader)):
        batch_inputs = batch_inputs.to(device, non_blocking=True)
        batch_labels = batch_labels.float().to(device, non_blocking=True)

        batch_outputs = model(batch_inputs)
        batch_loss = criterion(batch_outputs, batch_labels)

        val_epoch_loss += batch_loss.detach() / len(val_dataloader)

        batch_thresh_outputs = batch_outputs > threshold
        batch_score_val = score(batch_labels, batch_thresh_outputs)
    score_epoch_val = score.compute()

    return score_epoch_val, val_epoch_loss


def load_object(obj_path: str, default_obj_path: str = "") -> Any:
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f"Object `{obj_name}` cannot be loaded from `{obj_path}`.")
    return getattr(module_obj, obj_name)
