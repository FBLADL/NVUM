import os
from PIL import ImageDraw
import random

import numpy as np
from numpy.lib.type_check import _imag_dispatcher
import pandas as pd
import torch
from PIL import Image
from skimage import io
from sklearn.preprocessing import MultiLabelBinarizer
from torch.nn.modules import transformer
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import wandb, csv
from loguru import logger

# from utils.gcloud import download_chestxray_unzip

Labels = {
    "Atelectasis": 0,
    "Cardiomegaly": 1,
    "Effusion": 2,
    "Infiltration": 3,
    "Mass": 4,
    "Nodule": 5,
    "Pneumonia": 6,
    "Pneumothorax": 7,
    "Consolidation": 8,
    "Edema": 9,
    "Emphysema": 10,
    "Fibrosis": 11,
    "Pleural_Thickening": 12,
    "Hernia": 13,
    "No Finding": 14,
}
mlb = MultiLabelBinarizer(classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])


class ChestDataset(Dataset):
    def __init__(self, root_dir, transforms, mode, args) -> None:

        self.transform = transforms
        self.root_dir = root_dir
        self.mode = mode
        self.pathologies = np.array(list(Labels))
        df_path = os.path.join(root_dir, "Data_Entry_2017.csv")
        gt = pd.read_csv(df_path, index_col=0)
        gt = gt.to_dict()["Finding Labels"]

        if mode == "test":
            img_list = os.path.join(root_dir, "test_list.txt")
        elif mode == "clean_test":
            img_list = os.path.join(root_dir, "test_labels.csv")
        else:
            img_list = os.path.join(root_dir, "train_val_list.txt")

        if mode == "clean_test":
            gt = pd.read_csv(img_list, index_col=0)
            self.imgs = gt.index.to_numpy()
            gt = gt.iloc[:, -5:-1]
            # replace NO YES with 0 1
            gt = gt.replace("NO", 0)
            gt = gt.replace("YES", 1)
            clean_gt = gt.to_numpy()
            self.gt = np.zeros((clean_gt.shape[0], args.num_classes))
            # Pheumothorax
            self.gt[:, 7] = clean_gt[:, 1]
            # Nodule
            self.gt[:, 4] = clean_gt[:, 3]
            # Mass
            self.gt[:, 5] = clean_gt[:, 3]
        else:
            with open(img_list) as f:
                names = f.read().splitlines()

            self.imgs = np.asarray([x for x in names])
            gt = np.asarray([gt[i] for i in self.imgs])
            self.gt = np.zeros((gt.shape[0], 15))
            for idx, i in enumerate(gt):
                target = i.split("|")
                binary_result = mlb.fit_transform(
                    [[Labels[i] for i in target]]
                ).squeeze()
                self.gt[idx] = binary_result

            if args.trim_data:
                self.trim()
                # Assign no finding
                row_sum = np.sum(self.gt, axis=1)
                # self.gt[np.where(row_sum == 0), -1] = 1
                drop_idx = np.where(row_sum == 0)[0]
                self.gt = np.delete(self.gt, drop_idx, axis=0)
                self.imgs = np.delete(self.imgs, drop_idx, axis=0)

            self.label_distribution = self.calc_prior()
        # tmp_col = list(self.pathologies)  # list(Labels.keys()).copy()
        # tmp_col.extend(["MeanAUC-14c"])

        # f = open(os.path.join(wandb.run.dir, "pred.csv"), "w")
        # writer = csv.writer(f)
        # writer.writerow(tmp_col)
        # f.close
        # logger.bind(stage="DATA").info(f"{self.pathologies}")

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, "data", self.imgs[index])
        gt = self.gt[index]

        img = Image.fromarray(io.imread(img_path)).convert("RGB")
        img_t = self.transform(img)

        return img_t, gt, index

    def __len__(self):
        return self.imgs.shape[0]

    def calc_prior(self):
        tmp = []
        no_finding_prob = np.nonzero(self.gt[:, -1] == 1)[0].shape[0] / self.gt.shape[0]
        for i in range(self.gt.shape[1] - 1):
            tmp.append(
                (1 - no_finding_prob)
                * np.nonzero(self.gt[:, i] == 1)[0].shape[0]
                / self.gt.shape[0]
            )
        tmp.append(no_finding_prob)
        print(tmp)
        # print(self.gt.shape[0])

        return tmp

    def trim(self):
        cut_list = [
            # 'Mass',
            # 'Nodule',
            "Consolidation",
        ]

        for dp_class in cut_list:
            drop_idx_col = np.where(self.pathologies == dp_class)[0].item()
            drop_idx_row = np.where(self.gt[:, drop_idx_col] == 1.0)[0]
            if len(drop_idx_row) == 0:
                print(f"skip {dp_class}")
                continue
            self.pathologies = np.delete(self.pathologies, drop_idx_col)
            self.gt = np.delete(self.gt, drop_idx_row, axis=0)
            self.imgs = np.delete(self.imgs, drop_idx_row, axis=0)

            self.gt = np.delete(self.gt, drop_idx_col, axis=1)
        return


def construct_cx14_cut(args, root_dir, mode):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if mode == "train" or mode == "influence":
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (args.resize, args.resize), scale=(0.2, 1)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(args.resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    dataset = ChestDataset(root_dir, transform, mode, args)
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size if mode != "influence" else 1,
        shuffle=True if mode == "train" else False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )

    return loader, dataset.label_distribution if mode != "clean_test" else None
