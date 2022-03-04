import os, sys
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch
import numpy as np
import torch.nn as nn
from opts import parse_args
import torchvision
from models.densenet import densenet121

# from data.cx14_dataloader import construct_loader
from data.openi_dataloader_cut import construct_openi_cut as construct_loader
from loguru import logger
import wandb
from glob import glob
from utils import *

BRED = color.BOLD + color.RED


def config_wandb(args):
    EXP_NAME = "NIH-CX-14"
    os.environ["WANDB_MODE"] = args.wandb_mode
    # os.environ["WANDB_SILENT"] = "true"
    wandb.init(project=EXP_NAME)
    wandb.run.name = wandb.run.id
    config = wandb.config
    config.update(args)
    if args.wandb_mode == "online":
        code = wandb.Artifact("project-source", type="code")
        for path in glob("*.py", recursive=True):
            code.add_file(path)
        wandb.run.use_artifact(code)
    logger.bind(stage="CONFIG").critical("WANDB_MODE = {}".format(args.wandb_mode))
    logger.bind(stage="CONFIG").info("Experiment Name: {}".format(EXP_NAME))
    return


def load_args():
    args = parse_args()
    # args.batch_size = 16
    # args.num_workers = 8
    args.use_ensemble = False

    logger.bind(stage="CONFIG").critical(
        "use_ensemble = {}".format(str(args.use_ensemble))
    )
    return args


def main():
    BEST_AUC = -np.inf
    args = load_args()
    config_wandb(args)
    try:
        os.mkdir(args.save_dir)
    except OSError as error:
        logger.bind(stage="CONFIG").debug(error)

    model1 = densenet121(pretrained=True)
    model1.classifier = nn.Linear(1024, 15)
    model1.load_state_dict(torch.load("./ckpt/Baseline-MLSM.pth")["net"])
    # model1.classifier = nn.Identity
    # from models.model import model_disentangle
    # model1 = model_disentangle()

    model1.to(args.device)
    if args.use_ensemble:
        model2 = densenet121(pretrained=True)
        model2.classifier = nn.Linear(1024, 15)
        model2.to(args.device)
    optim1 = torch.optim.Adam(
        model1.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0, eps=0.1
    )
    if args.use_ensemble:
        optim2 = torch.optim.Adam(
            model2.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0, eps=0.1
        )

    # wandb.watch(model1, log="all")

    train_loader = construct_loader(args, args.root_dir, "train")
    test_loader = construct_loader(args, args.root_dir, "test")

    scaler = torch.cuda.amp.GradScaler(enabled=True)
    criterion = nn.MultiLabelSoftMarginLoss().to(args.device)

    logger.bind(stage="TRAIN").info("Start Training")
    lr = args.lr

    all_auc, test_loss = test(
        criterion,
        model1,
        test_loader,
        args.num_classes,
        args.device,
    )
    mean_auc = np.asarray(all_auc).mean()

    wandb.log({"Test Loss": test_loss, "MeanAUC_14c": mean_auc})

    logger.bind(stage="EVAL").success(
        f"Test Loss {test_loss:0.4f} Mean AUC {mean_auc:0.4f}"
    )
    return


def train(scaler, criterion, net, optimizer, train_loader, device):
    net.train()
    total_loss = 0.0
    with tqdm(train_loader, desc="Train", ncols=100) as tl:
        for batch_idx, (inputs, labels, item) in enumerate(tl):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=True):
                outputs = net(inputs)
                loss = criterion(outputs, labels)
            total_loss += loss.item()
            tl.set_description_str(desc=BRED + f"Loss {loss.item():0.4f}" + color.END)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            lr_value = optimizer.param_groups[0]["lr"]
            wandb.log({"Learning Rate": lr_value, "MultiLabelSoftMarginLoss": loss})

    return total_loss / (batch_idx + 1)


def test(criterion, net, test_loader, num_classes, device, net2=None):
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
            if net2 is not None:
                outputs2 = net2(inputs)
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


def test_openi(args, model, model2):
    logger.bind(stage="EVAL").info("************** EVAL ON OPENI **************")
    # wandb.watch(model, log="all")

    # train_loader = construct_loader(args, args.root_dir, "train")
    test_loader = construct_loader(args, args.openi_root_dir, "test")

    # scaler = torch.cuda.amp.GradScaler(enabled=True)
    criterion = nn.MultiLabelSoftMarginLoss().to(args.device)

    logger.bind(stage="TRAIN").info("Start Training")
    lr = args.lr

    all_auc, test_loss = test(
        criterion,
        model,
        test_loader,
        args.num_classes,
        args.device,
    )

    mean_auc = np.asarray(all_auc).mean()
    wandb.log({"Test Loss OPENI": test_loss, "MeanAUC_14c OPENI": mean_auc})
    logger.bind(stage="EVAL").success(
        f"Test Loss {test_loss:0.4f} Mean AUC {mean_auc:0.4f}"
    )
    return all_auc, mean_auc


if __name__ == "__main__":

    fmt = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS} </green> | <bold><cyan> [{extra[stage]}] </cyan></bold> | <level>{level: <8}</level> | <level>{message}</level>"
    logger.remove()
    logger.add(sys.stderr, format=fmt)
    main()
