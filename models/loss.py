"""
Author: Nasir Hayat (nasirhayat6160@gmail.com)
Date: June 10, 2020
"""

from __future__ import print_function

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import linalg as LA
from torch.nn.functional import kl_div, softmax, log_softmax


class ASELR(nn.Module):
    def __init__(
        self, num_samples, num_classes=15, lam=3, beta=0.9, device=0, prior=None, tau=1
    ) -> None:
        super(ASELR, self).__init__()
        self.num_samples = num_samples
        self.pred_hist = torch.zeros(num_samples, num_classes).to(device)
        self.pred_hist_neg = torch.zeros(num_samples, num_classes).to(device)
        self.beta = beta
        self.lam = lam
        self.prior = torch.from_numpy(np.asarray(prior)).to(device)
        self.tau = tau

    def update_hist(self, epoch, out, target, index=None, mix_index=..., mixup_l=1):
        # if self.prior is not None:
        #     adjust_logits = self.logits_adjust(out)
        # else:
        #     adjust_logits = out
        y_pred_ = torch.sigmoid(out).float()
        y_pred_neg = 1 - y_pred_
        # y_pred_neg = (y_pred_neg + 0.01).clamp(max=1)
        # TODO negative probability shifting
        self.pred_hist[index] = (
            self.beta * self.pred_hist[index] + (1 - self.beta) * y_pred_
        )
        self.pred_hist_neg[index] = (
            self.beta * self.pred_hist_neg[index] + (1 - self.beta) * y_pred_neg
        )

        # self.pred_hist_neg[index] = (self.pred_hist_neg[index] - 0.05).clamp(min=0)

        self.q = (
            mixup_l * self.pred_hist[index]
            + (1 - mixup_l) * self.pred_hist[index][mix_index]
        )
        self.q_neg = (
            mixup_l * self.pred_hist_neg[index]
            + (1 - mixup_l) * self.pred_hist_neg[index][mix_index]
        )

    def forward(self, output, y_labeled, index=None):
        targets = self.q if index is None else self.pred_hist[index.item()]
        targets_neg = self.q_neg if index is None else self.pred_hist_neg[index.item()]

        y_pred = torch.sigmoid(output).clamp(max=1.0 - 1e-4)
        y_pred_neg = 1 - y_pred
        # y_pred_neg = (y_pred_neg + 0.01).clamp(max=1)

        reg = (
            1
            - (
                1 * targets * y_pred * (1 - y_labeled)
                + 3 * targets_neg * y_pred_neg * y_labeled
            )
        ).log()
        reg_pos = (1 - targets * y_pred).log()
        reg_neg = (1 - targets_neg * y_pred_neg).log()
        # reg_neg = (1 - targets_neg * y_pred_neg).log()
        # reg = reg_neg
        wandb.log({"Reg Pos": reg_pos.mean().item(), "Reg Neg": reg_neg.mean().item()})

        # torch.set_grad_enabled(False)
        # pt0 = y_pred * y_labeled
        # pt1 = y_pred_neg * (1 - y_labeled)
        # pt = pt0 + pt1
        # # TODO switch frocus on positive to negative
        # # TODO correct! keep postive gamma larger than negative
        # one_sided_gamma = 3 * y_labeled + 1 * (1 - y_labeled)
        # one_sided_w = torch.pow(pt, one_sided_gamma)
        # torch.set_grad_enabled(True)
        # reg *= one_sided_w
        # y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)
        bce_loss = nn.BCEWithLogitsLoss(reduction="none")(output, y_labeled)

        # reg = (1 - (self.q * y_pred)).log()
        return bce_loss, reg
        # return bce_loss_pos, bce_loss_neg, reg_pos, reg_neg


# TODO calculate gradient norm for uncertainty of label before updating early targets
# TODO Gradient analysis for imbalanced follow ELR approach.
class ELR(nn.Module):
    def __init__(
        self, num_samples, num_classes=15, lam=3, beta=0.9, device=0, prior=None, tau=1
    ) -> None:
        super(ELR, self).__init__()
        self.num_samples = num_samples
        self.pred_hist = torch.zeros(num_samples, num_classes).to(device)
        self.beta = beta
        self.lam = lam
        self.prior = torch.from_numpy(np.asarray(prior)).to(device)
        self.tau = tau

    def logits_adjust(self, logits):
        adjust_term = (self.prior.unsqueeze(0) + 1e-12).log() * self.tau
        adjust_term = adjust_term.detach()
        # print(logits.shape)
        # print(adjust_term.shape)
        return logits - adjust_term

    def logits_sub(self, logits):
        adjust_term = (self.prior.unsqueeze(0) + 1e-12).log() * self.tau
        adjust_term = adjust_term.detach()
        # print(logits.shape)
        # print(adjust_term.shape)
        return logits - adjust_term

    def forward(self, output, y_labeled, index=None):
        # if self.prior is not None:
        #     adjust_logits = self.logits_adjust(output)
        # else:
        #     adjust_logits = output
        y_pred = torch.sigmoid(output)

        y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)

        bce_loss = nn.BCEWithLogitsLoss(reduction="none")(output, y_labeled)
        if index is not None:
            reg = (1 - (self.pred_hist[index.item()] * y_pred)).log()
        else:
            # reg = (1 - (torch.sigmoid(self.q) * y_pred)).log()
            reg = (1 - torch.sigmoid(self.q * y_pred.detach())).log()
        # final_loss = torch.mean(bce_loss + self.lam * reg)

        return bce_loss, reg

    def update_hist(self, epoch, out, target, index=None, mix_index=..., mixup_l=1):
        if self.prior is not None:
            adjust_logits = self.logits_adjust(out).float()
        else:
            adjust_logits = out.float()

        # adjust_logits = out
        # y_pred_ = torch.sigmoid(out).float()
        self.pred_hist[index] = (
            self.beta * self.pred_hist[index] + (1 - self.beta) * adjust_logits
        )
        self.q = (
            mixup_l * self.pred_hist[index]
            + (1 - mixup_l) * self.pred_hist[index][mix_index]
        )

    def load_hist(self, hist):
        self.pred_hist = hist
