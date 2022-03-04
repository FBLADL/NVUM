import numpy as np
import os
import torch
import wandb
import csv
from loguru import logger


class color:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def print_auroc(aurocIndividual):
    aurocIndividual = np.round(aurocIndividual, 4)

    MeanAUC_15c = np.round(aurocIndividual[0:].mean(), 4)
    MeanAUC_14c = np.round(aurocIndividual[1:].mean(), 4)

    row_csv = list(aurocIndividual).copy()
    row_csv.extend([MeanAUC_15c, MeanAUC_14c])
    # self.df_result.iloc[epoch] = list(aurocIndividual).extend([MeanAUC_15c, MeanAUC_14c])

    f = open(os.path.join(wandb.run.dir, "pred.csv"), "a")
    writer = csv.writer(f)
    writer.writerow(row_csv)
    f.close

    wandb.log({"MeanAUC_14c": MeanAUC_14c})
    logger.bind(stage="TEST").success(
        "| MeanAUC_15c: {} | MeanAUC_14c: {} |".format(MeanAUC_15c, MeanAUC_14c)
    )
    return MeanAUC_14c


def save_checkpoint(state_dict, epoch, save_dir, is_best=False):
    # path = os.path.join(wandb.run.dir, "model_{}.pth".format(prefix))
    # torch.save(state_dict, os.path.join(wandb.run.dir, "model_{}.pth".format(epoch)))
    if is_best:
        torch.save(state_dict, os.path.join(save_dir))
        logger.bind(stage="EVAL").critical(f"Saving best checkpoint")
        torch.save(state_dict, os.path.join(wandb.run.dir, "model_best.pth"))
    else:
        torch.save(state_dict, os.path.join(save_dir, f"model_{epoch}.pth"))
        logger.bind(stage="EVAL").critical(f"Saving {epoch} checkpoint")


def log_csv(epoch, all_auc, mean_auc, path):
    tmp_row = all_auc.copy()
    tmp_row.extend([mean_auc])
    tmp_row.insert(0, [epoch])
    # f = open(os.path.join(wandb.run.dir, "pred.csv"), "a")
    f = open(path, "a")
    writer = csv.writer(f)
    writer.writerow(tmp_row)
    f.close
