import argparse

import torch
import random
import numpy as np
from clearml import Logger, Task
from torchmetrics.classification import MultilabelFBetaScore

from data.dataloader import load_data
from src.utils.config import Config
from src.utils.train_utils import load_object
from src.model.model import LandscapeClassifier
from utils.train_utils import train_one_epoch, validate_one_epoch


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file")
    return parser.parse_args()


def train_model(model, dataloaders, config):

    seed = config.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    log = Logger.current_logger()
    # Create a temporary directory to save training checkpoints

    f_beta = MultilabelFBetaScore(beta=config.beta, num_labels=config.n_classes).to(
        device
    )
    criterion = load_object(conf.criterion)(
        **conf.criterion_kwargs,
    )
    optimizer = load_object(conf.optimizer)(
        model.parameters(),
        **conf.optimizer_kwargs,
    )
    scheduler = load_object(conf.scheduler)(optimizer, **conf.scheduler_kwargs)

    best_fbeta_score = 0
    for epoch in range(config.num_epochs):
        # Each epoch has a training and validation phase
        score_train_epoch, avg_loss_train = train_one_epoch(
            model,
            dataloaders["train"],
            config.threshold,
            f_beta,
            criterion,
            optimizer,
            device,
        )
        with torch.no_grad():
            score_val_epoch, avg_loss_val = validate_one_epoch(
                model,
                dataloaders["val"],
                config.threshold,
                f_beta,
                criterion,
                device,
            )

            # saving values for debugging
        if score_val_epoch > best_fbeta_score:
            best_fbeta_score = score_val_epoch
            torch.save(model, config.name_model + "_fold.pth")

            log.report_single_value(name='Best Fbeta score', value=best_fbeta_score)
            log.report_single_value(name='Best loss', value=avg_loss_val)

        log.report_scalar("loss", "Test", iteration=epoch, value=avg_loss_train)
        log.report_scalar("loss", "Train", iteration=epoch, value=avg_loss_val)

        log.report_scalar("Fbeta score", "Test", iteration=epoch, value=score_train_epoch)
        log.report_scalar("Fbeta score", "Train", iteration=epoch, value=score_val_epoch)

        scheduler.step()

    return model


if __name__ == "__main__":
    args = arg_parse()
    conf = Config.from_yaml(args.config_file)

    device = torch.device(conf.device)

    prev_task = Task.get_task(project_name='hw01-modeling', task_name=conf.name_model)

    dataloaders = load_data(conf)

    model_ft = LandscapeClassifier(conf)
    model_ft = model_ft.to(device)

    model = train_model(
        model_ft,
        dataloaders,
        config=conf,
    )
    prev_task.close()
