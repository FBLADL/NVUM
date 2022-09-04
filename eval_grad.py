import torch
import os
from torch.autograd import grad
from data.cx14_dataloader_cut import construct_cx14_cut as construct_cx14_loader
from opts import parse_args
from models.densenet import densenet121
from models.loss import NVUMREG
from train import log_init


def main():
    args = parse_args()
    log_pack = log_init(args)
    influence_loader = construct_cx14_loader(args, args.train_root_dir, "influence")

    net = densenet121()
    state_dict = torch.load(os.path.join(log_pack["cks"], f"model{args.resume}.pth"))
    net.load_state_dict(state_dict["net1"])
    criterion = NVUMREG(
        len(influence_loader.dataset),
        args.num_classes,
        lam=1,
        beta=0.9,
        prior=None,
        tau=1,
    )
    criterion.load_hist(state_dict["elt1"])

    get_grad(
        influence_loader, net, criterion, args.device, log_pack["grad"], args.resume
    )


def get_grad(train_loader, net, criterion, device, save_path, epoch):
    grad_bce_list = []
    grad_reg_list = []
    net.eval()
    params = [p for p in net.parameters() if p.requires_grad and len(p.size()) != 1]

    for batch_idx, (inputs, labels, item) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        preds = net(inputs)
        bce_loss, reg = criterion(preds, labels)
        grad_bce = grad(bce_loss, params)
        grad_reg = grad(reg, params)

        for ele_bce, ele_reg in zip(grad_bce, grad_reg):
            ele_bce.detach()
            ele_reg.detach()
            grad_bce_list.append(ele_bce)
            grad_reg_list.append(ele_reg)

        for i in range(len(grad_bce[0])):
            per_param_bce_grad = [
                torch.unsqueeze(per_img_grad[i], dim=0)
                for per_img_grad in grad_bce_list
            ]
            per_param_reg_grad = [
                torch.unsqueeze(per_img_grad[i], dim=0)
                for per_img_grad in grad_reg_list
            ]

            per_param_bce_grad = torch.cat(per_param_bce_grad, dim=0)
            per_param_reg_grad = torch.cat(per_param_reg_grad, dim=0)

            per_param_bce_grad = torch.sum(per_param_bce_grad, dim=0)
            per_param_reg_grad = torch.sum(per_param_reg_grad, dim=0)

            save_bce_path = os.path.join(save_path, f"grad_bce{epoch}.pt")
            save_reg_path = os.path.join(save_path, f"grad_reg_{epoch}.pt")

            torch.save(per_param_bce_grad, save_bce_path)
            torch.save(per_param_reg_grad, save_reg_path)
