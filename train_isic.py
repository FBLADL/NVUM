import os, sys
from pathlib import Path
from cv2 import CAP_PROP_XI_LENS_FOCAL_LENGTH
from torch.autograd import grad
from tqdm import tqdm
from torch.autograd import grad
from sklearn.metrics import roc_auc_score
import torch
import numpy as np
import torch.nn as nn
from opts import parse_args
from models.densenet import densenet121, NormalizedLinear
from torchvision.models.resnet import resnet18
from models.loss import ELR_ISIC as ELR
from data.cx14_dataloader_cut import construct_cx14_cut as construct_cx14_loader
from data.cxp_dataloader_cut import construct_cxp_cut as construct_cxp_loader
from data.isic_dataloader import construct_isic

# from data.openi import construct_loader
from loguru import logger
import wandb
from utils import *
from eval_openi import test_openi
from eval_pdc import test_pc

# from eval_grad import get_grad

BRED = color.BOLD + color.RED
nih_stored_trim_list = "epoch,Atelectasis,Cardiomegaly,Effusion,Infiltration,Mass,Nodule,Pneumonia,Pneumothorax,Edema,Emphysema,Fibrosis,Pleural_Thickening,Hernia,Mean\n"


def linear_rampup(current, rampup_length=10):
    current = np.clip((current) / rampup_length, 0.0, 1.0)
    return float(current)


def config_wandb(args):
    EXP_NAME = args.exp_name
    os.environ["WANDB_MODE"] = args.wandb_mode
    # os.environ["WANDB_SILENT"] = "true"
    wandb.init(project=EXP_NAME)
    wandb.run.name = args.run_name
    # wandb.run.dir = os.path.join(args.save_dir, args.run_name)
    config = wandb.config
    config.update(args)
    logger.bind(stage="CONFIG").critical("WANDB_MODE = {}".format(args.wandb_mode))
    logger.bind(stage="CONFIG").info("Experiment Name: {}".format(EXP_NAME))


def load_args():
    args = parse_args()
    return args


def log_init(args):
    log_base = os.path.join(args.save_dir, args.run_name)

    ck_log = os.path.join(log_base, "cks")
    Path(ck_log).mkdir(parents=True, exist_ok=True)

    grad_log = os.path.join(log_base, "grads")
    Path(grad_log).mkdir(parents=True, exist_ok=True)

    best_ck_log = os.path.join(log_base, "model_best.pth")
    info_log = os.path.join(log_base, "info.log")
    open(info_log, "a")
    logger.add(info_log, enqueue=True)

    train_csv = os.path.join(log_base, f"pred_{args.train_data}.csv")
    with open(train_csv, "a") as f:
        if args.trim_data:
            f.write(nih_stored_trim_list)

    openi_csv = os.path.join(log_base, "pred_openi.csv")
    with open(openi_csv, "a") as f:
        if args.trim_data:
            f.write(nih_stored_trim_list)

    pd_csv = os.path.join(log_base, "pred_padchest.csv")
    with open(pd_csv, "a") as f:
        if args.trim_data:
            f.write(nih_stored_trim_list)

    return {
        "cks": ck_log,
        "info": info_log,
        "train_csv": train_csv,
        "openi_csv": openi_csv,
        "pd_csv": pd_csv,
        "best_ck": best_ck_log,
        "grad": grad_log,
    }


def main():
    BEST_AUC = -np.inf
    global args
    args = load_args()
    log_pack = log_init(args)
    config_wandb(args)

    model1, model1_ema = create_model_ema(densenet121, 8, args.device)
    optim1, optim1_ema = create_optimizer_ema(model1, model1_ema, args)

    wandb.watch(model1, log="all")

    (
        train_loader,
        test_loader,
        eval_train_loader,
        train_label_distribution,
    ) = construct_isic("/run/media/Data/ISIC2019/")

    scaler = torch.cuda.amp.GradScaler(enabled=True)
    # criterion = nn.MultiLabelSoftMarginLoss().to(args.device)
    criterion1 = ELR(
        len(train_loader.dataset),
        num_classes=8,
        device=args.device,
        beta=args.reg_update_beta,
        prior=train_label_distribution,
    )
    logger.bind(stage="TRAIN").info("Start Training")
    lr = args.lr

    np.save("clean.npy", np.array(eval_train_loader.dataset.clean_targets))
    np.save("noisy.npy", eval_train_loader.dataset.noise_targets)
    # test_openi(args, model=model1_ema, model2=model2_ema if args.use_ensemble else None)
    for epoch in range(args.total_epochs):
        if epoch == (0.7 * args.total_epochs) or epoch == (0.9 * args.total_epochs):
            lr *= 0.1
        for param in optim1.param_groups:
            param["lr"] = lr
        train_loss1, acc, mem_acc, incorrect_acc = train(
            scaler,
            args,
            epoch,
            criterion1,
            model1,
            model1_ema,
            optim1,
            optim1_ema,
            train_loader,
            args.device,
        )
        # print(acc, mem_acc, incorrect_acc)
        all_differences = eval_train(
            args, epoch, model1, criterion1, optim1, eval_train_loader, args.device
        )
        np.save(f"diff{epoch}.npy", all_differences.numpy())
        # print(acc, mem_acc, incor_acc)
        # ce_grads, reg_grads = eval_train(
        #     args, epoch, model1, criterion1, optim1, eval_train_loader, args.device
        # )
        # all_grads = ce_grads + reg_grads
        # np.save(f"grads{epoch}.npy", all_grads)
        train_loss = train_loss1
        all_acc, test_loss = test(
            model1_ema,
            test_loader,
            args.num_classes,
            args.device,
        )
        logger.bind(stage="EVAL").success(f"{all_acc}")


def eval_train(args, epoch, net, criterion, optimizer, train_loader, device):
    net.eval()
    all_ce_grads = []
    all_reg_grads = []
    correct, incorrect, mem = 0, 0, 0
    total_incorrect = 0
    all_differences = torch.tensor([])
    for batch_idx, (inputs, noise_label, clean_label, item) in enumerate(
        tqdm(train_loader)
    ):
        inputs, clean_labels, labels = (
            inputs.to(device),
            clean_label.to(device),
            noise_label.to(device),
        )
        optimizer.zero_grad()
        outputs = net(inputs)
        probs = torch.softmax(outputs, dim=1)
        all_differences = torch.cat(
            [
                all_differences,
                torch.hstack([1 - probs[i][labels[i]] for i in range(probs.shape[0])])
                .detach()
                .cpu(),
            ],
            dim=0,
        )
        # ce_loss, reg = criterion(outputs, clean_labels)
        # ce_loss, reg = torch.mean(ce_loss), torch.mean(reg)
        # grad_ce = grad(ce_loss, net.parameters(), retain_graph=True)[-1].mean().item()
        # grad_ce = np.array(
        #     [
        #         i.mean().item()
        #         for i in grad(ce_loss, net.parameters(), retain_graph=True)
        #     ]
        # ).mean()
        # grad_reg = grad(reg, net.parameters(), retain_graph=True)[-1].mean().item()
        # grad_reg = np.array(
        #     [i.mean().item() for i in grad(reg, net.parameters(), retain_graph=True)]
        # ).mean()
        # all_ce_grads.append(grad_ce)
        # all_reg_grads.append(grad_reg)
    #     _, pred = outputs.max(1)
    #     total_incorrect += (clean_labels.to(device) != labels).nonzero().shape[0]
    #     correct += (
    #         pred[(clean_labels.to(device) != labels).nonzero()]
    #         .squeeze()
    #         .eq(
    #             clean_labels.to(device)[
    #                 (clean_labels.to(device) != labels).nonzero()
    #             ].squeeze()
    #         )
    #         .sum()
    #         .item()
    #     )
    #     mem += (
    #         pred[(clean_labels.to(device) != labels).nonzero()]
    #         .squeeze()
    #         .eq(labels[(clean_labels.to(device) != labels).nonzero()].squeeze())
    #         .sum()
    #         .item()
    #     )
    #     incorrect += (
    #         (clean_labels.to(device) != labels).nonzero().shape[0]
    #         - (
    #             pred[(clean_labels.to(device) != labels).nonzero()]
    #             .squeeze()
    #             .eq(
    #                 clean_labels.to(device)[
    #                     (clean_labels.to(device) != labels).nonzero()
    #                 ].squeeze()
    #             )
    #             .sum()
    #             .item()
    #         )
    #         - pred[(clean_labels.to(device) != labels).nonzero()]
    #         .squeeze()
    #         .eq(labels[(clean_labels.to(device) != labels).nonzero()].squeeze())
    #         .sum()
    #         .item()
    #     )
    # total_num = total_incorrect
    # return (
    #     correct / total_num,
    #     mem / total_num,
    #     incorrect / total_num,
    # )
    return all_differences


def train(
    scaler,
    args,
    epoch,
    criterion,
    net,
    net_ema,
    optimizer,
    optimizer_ema,
    train_loader,
    device,
):
    net.train()
    net_ema.train()
    total_loss = 0.0
    correct = 0
    mem = 0
    incorrect = 0
    total_incorrect = 0
    with tqdm(train_loader, desc="Train", ncols=100) as tl:
        for batch_idx, (inputs, labels, clean_labels, item) in enumerate(tl):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            lam = np.random.beta(1.0, 1.0)
            lam = max(lam, 1 - lam)
            mix_index = torch.randperm(inputs.shape[0]).to(device)

            with torch.cuda.amp.autocast(enabled=True):
                outputs = net(inputs)
                outputs_ema = net_ema(inputs).detach()
                _, pred = outputs_ema.max(1)
                total_incorrect += (
                    (clean_labels.to(device) != labels).nonzero().shape[0]
                )
                correct += (
                    pred[(clean_labels.to(device) != labels).nonzero()]
                    .squeeze()
                    .eq(
                        clean_labels.to(device)[
                            (clean_labels.to(device) != labels).nonzero()
                        ].squeeze()
                    )
                    .sum()
                    .item()
                )
                mem += (
                    pred[(clean_labels.to(device) != labels).nonzero()]
                    .squeeze()
                    .eq(labels[(clean_labels.to(device) != labels).nonzero()].squeeze())
                    .sum()
                    .item()
                )
                incorrect += (
                    (clean_labels.to(device) != labels).nonzero().shape[0]
                    - (
                        pred[(clean_labels.to(device) != labels).nonzero()]
                        .squeeze()
                        .eq(
                            clean_labels.to(device)[
                                (clean_labels.to(device) != labels).nonzero()
                            ].squeeze()
                        )
                        .sum()
                        .item()
                    )
                    - pred[(clean_labels.to(device) != labels).nonzero()]
                    .squeeze()
                    .eq(labels[(clean_labels.to(device) != labels).nonzero()].squeeze())
                    .sum()
                    .item()
                )

                criterion.update_hist(
                    epoch,
                    outputs_ema,
                    labels.float(),
                    item.numpy().tolist(),
                    mix_index=mix_index,
                    mixup_l=lam,
                )

                bce_loss, reg = criterion(outputs, labels)
                final_loss = torch.mean(bce_loss + args.reg_weight * reg)
            total_loss += final_loss.item()
            tl.set_description_str(
                desc=BRED
                + f"BCE {bce_loss.mean().item():0.4f} Reg {reg.mean().item():.4f} Final {final_loss.item():.4f}"
                + color.END
            )
            scaler.scale(final_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            optimizer_ema.step()
            lr_value = optimizer.param_groups[0]["lr"]
            wandb.log(
                {
                    "MultiLabelSoftMarginLoss": bce_loss.mean().item(),
                    "Reg": reg.mean().item(),
                }
            )
            # break

    total_num = total_incorrect
    return (
        total_loss / (batch_idx + 1),
        correct / total_num,
        mem / total_num,
        incorrect / total_num,
    )


def test(net, test_loader, num_classes, device, net2=None, clean_test=False):
    logger.bind(stage="EVAL").info("************** EVAL ON NIH **************")
    net.eval()
    all_preds = torch.FloatTensor([]).to(device)
    all_gts = torch.FloatTensor([]).to(device)
    total_loss = 0.0
    correct = 0
    for batch_idx, (inputs, labels, _, item) in enumerate(
        tqdm(test_loader, desc="Test       ", ncols=100)
    ):
        with torch.no_grad():
            inputs, labels = inputs.to(device), labels.to(device)

            outputs1 = net(inputs)
            outputs = outputs1

            loss = nn.CrossEntropyLoss()(outputs, labels)
            # loss = nn.BCEWithLogitsLoss()(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.softmax(outputs, dim=1).max(1)

            correct += preds.eq(labels).sum().item()

            all_preds = torch.cat((all_preds, preds), dim=0)
            all_gts = torch.cat((all_gts, labels), dim=0)

    return correct / len(test_loader.dataset), total_loss / (batch_idx + 1)
    return all_auc, total_loss / (batch_idx + 1)


def create_model_ema(arch, num_classes, device):
    model = resnet18(pretrained=True)
    model.fc = nn.Linear(512, num_classes)

    model_ema = resnet18(pretrained=True)
    model_ema.fc = nn.Linear(512, num_classes)
    for param in model_ema.parameters():
        param.detach_()

    return model.to(device), model_ema.to(device)


def create_optimizer_ema(model, model_ema, args):
    optim = torch.optim.Adam(
        list(filter(lambda p: p.requires_grad, model.parameters())),
        lr=args.lr,
        betas=(0.9, 0.99),
        eps=0.1,
    )
    optim_ema = WeightEMA(model, model_ema)
    for param in model_ema.parameters():
        param.detach_()

    return optim, optim_ema


class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.99):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        # self.params = model.module.state_dict()
        # self.ema_params = ema_model.module.state_dict()
        self.params = model.state_dict()
        self.ema_params = ema_model.state_dict()
        # self.wd = 0.02 * args.lr

        for (k, param), (ema_k, ema_param) in zip(
            self.params.items(), self.ema_params.items()
        ):
            ema_param.data.copy_(param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for (k, param), (ema_k, ema_param) in zip(
            self.params.items(), self.ema_params.items()
        ):
            if param.type() == "torch.cuda.LongTensor":
                ema_param = param
            else:
                # if "num_batches_tracked" in k:
                #     ema_param.copy_(param)
                # else:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)


if __name__ == "__main__":

    fmt = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS} </green> | <bold><cyan> [{extra[stage]}] </cyan></bold> | <level>{level: <8}</level> | <level>{message}</level>"
    logger.remove()
    logger.add(sys.stderr, format=fmt)
    main()
