import os, sys
from torch._C import device
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch
import numpy as np
import torch.nn as nn
from opts import parse_args
import torchvision
from models.densenet import densenet121
from data.cx14_dataloader_cut import construct_cx14_cut
from data.openi_dataloader_cut import construct_openi_cut
from data.padchest_dataloader_cut import construct_pc_cut

# from data.openi import construct_loader
from loguru import logger
import wandb
from glob import glob
from utils import *

# from loss.SD_Loss import SDLLoss
# from loss.sd_loss_org import SDLLoss
# from models.model import model_zs_sdl
# from biobert import BERTEmbedModel
# from train_pd import TRAIN_PD
# from train_cls import TRAIN_CLS
from numpy import linalg as LA

# from helper_functions import get_knns, calc_F1
from torch.utils.data import Dataset, DataLoader
import copy

BRED = color.BOLD + color.RED


def config_wandb(args):
    EXP_NAME = "NIH-CX-(14)"
    os.environ["WANDB_MODE"] = args.wandb_mode
    # os.environ["WANDB_SILENT"] = "true"
    wandb.init(project=EXP_NAME, notes=args.run_note)
    wandb.run.name = wandb.run.id
    config = wandb.config
    config.update(args)
    # if args.wandb_mode == "online":
    #     code = wandb.Artifact("project-source", type="code")
    #     for path in glob("*.py", recursive=True):
    #         code.add_file(path)
    #     wandb.run.use_artifact(code)
    logger.bind(stage="CONFIG").critical("WANDB_MODE = {}".format(args.wandb_mode))
    logger.bind(stage="CONFIG").info("Experiment Name: {}".format(EXP_NAME))
    return


def load_args():
    args = parse_args()
    args.batch_size = 16
    args.num_workers = 8
    args.use_ensemble = False
    args.num_classes = 14
    args.lr_pd = 1e-4
    args.lr_cls = 0.05

    args.total_epochs = 40
    args.total_runs = 1
    args.num_pd = 7
    args.trim_data = True
    args.wandb_mode = "offline"

    args.run_note = "SAMPLE GRAPH RELABEL (0.3)"

    args.pc_root_dir = "/run/media/Data/"

    logger.bind(stage="CONFIG").critical(
        f"use_ensemble = {str(args.use_ensemble)} || num_pd = {args.num_pd}"
    )
    return args


def load_word_vec(args):
    wordvec_array = np.load("./embeddings/nih_biober_14_custom.npy")

    normed_wordvec = np.stack(
        [
            wordvec_array[i] / LA.norm(wordvec_array[i])
            for i in range(wordvec_array.shape[0])
        ]
    )
    normed_wordvec = (
        (torch.tensor(normed_wordvec).to(args.device).float())
        .permute(1, 0)
        .unsqueeze(0)
    )
    return normed_wordvec


def get_word_vec(model, pathology):
    tmp_list = []
    for i in pathology:
        current = model.get_embeds(i)
        tmp_list.append(current)
    saved_embd = torch.stack(tmp_list).numpy()
    return saved_embd


def _test(net, test_loader, num_classes, device, net2=None, prior=None):
    criterion = nn.BCEWithLogitsLoss()
    net.eval()
    if net2 is not None:
        net2.eval()
    all_preds = torch.FloatTensor([]).to(device)
    all_gts = torch.FloatTensor([]).to(device)
    total_loss = 0.0
    for batch_idx, (inputs, labels, item) in enumerate(
        tqdm(test_loader, desc="Test       ", ncols=100)
    ):
        with torch.no_grad():
            inputs, labels = inputs.to(device), labels.to(device)

            outputs1 = net(inputs)
            if prior is not None:
                outputs1 = outputs1
            if net2 is not None:
                outputs2 = net2(inputs)
                if prior is not None:
                    outputs2 = outputs2
                outputs = (outputs1 + outputs2) / 2
            else:
                outputs = outputs1

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.sigmoid(outputs)

            all_preds = torch.cat((all_preds, preds), dim=0)
            all_gts = torch.cat((all_gts, labels), dim=0)

    all_preds = all_preds.cpu().numpy()
    all_gts = all_gts.cpu().numpy()
    all_auc = [
        roc_auc_score(all_gts[:, i], all_preds[:, i]) for i in range(num_classes - 1)
    ]
    return all_auc, total_loss / (batch_idx + 1)


def test_pc(args, model, model2=None):
    logger.bind(stage="EVAL").info("************** EVAL ON PADCHEST **************")

    test_loader, pc_prior = construct_pc_cut(args, args.pc_root_dir, "test")

    all_auc, test_loss = _test(
        model,
        test_loader,
        args.num_classes,
        args.device,
        # word_vecs=wordvec_array,
        net2=None,
        prior=pc_prior,
    )

    mean_auc = np.asarray(all_auc).mean()
    wandb.log({"Test Loss PC": test_loss, "MeanAUC_14c PC": mean_auc})
    logger.bind(stage="EVAL").success(
        f"Test Loss {test_loss:0.4f} Mean AUC {mean_auc:0.4f}"
    )
    return all_auc, mean_auc


def main():
    BEST_AUC = -np.inf
    args = load_args()
    config_wandb(args)
    try:
        os.mkdir(args.save_dir)
    except OSError as error:
        logger.bind(stage="CONFIG").debug(error)

    # train_loader = construct_cx14(args, args.root_dir, "train")
    # test_loader = construct_cx14(args, args.root_dir, "test")
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    # wordvec_array = load_word_vec(args)
    # train_loader = construct_cx14_cut(args, args.root_dir, mode="train", file_name="train")
    # train_loader.dataset.gt = np.load("./archives/relabelled_gt_v1.npy")

    model = densenet121(pretrained=True)
    model.classifier = nn.Linear(1024, args.num_classes)
    # 88.07
    # model.load_state_dict(
    #     torch.load("./ckpt/run-20220117_223833-25jfss9m/files/model_best_cls.pth")[
    #         "net"
    #     ]
    # )
    # Baseline
    model.load_state_dict(
        torch.load("/mnt/hd/Logs/noisy_multi_label/log_adjust_3090/cks/model_23.pth")[
            "net1_ema"
        ]
    )

    model.to(args.device)
    criterion = nn.MultiLabelSoftMarginLoss().to(args.device)

    all_auc_pc, mean_auc_pc = test_pc(
        args,
        model,
    )
    log_csv(
        all_auc_pc,
        mean_auc_pc,
    )

    return


if __name__ == "__main__":

    fmt = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS} </green> | <bold><cyan> [{extra[stage]}] </cyan></bold> | <level>{level: <8}</level> | <level>{message}</level>"
    logger.remove()
    logger.add(sys.stderr, format=fmt)
    main()
