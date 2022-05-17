from __future__ import print_function

import wandb
import torch
import torch.nn as nn
import numpy as np

# from torch.nn.functional import kl_div, softmax, log_softmax
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
            reg = (1 - torch.sigmoid(self.q * y_pred.detach())).log()

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
