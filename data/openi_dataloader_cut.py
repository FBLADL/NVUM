import collections
import os
import os.path
import pprint
import random
import sys
import tarfile
import warnings
import zipfile

import imageio
import numpy as np
import pandas as pd
import pydicom
import skimage
import skimage.transform
from skimage.io import imread
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import xml
from loguru import logger
from PIL import Image
from skimage import io


def isNaN(string):
    return string != string


class Openi_Dataset(Dataset):
    """OpenI Dataset

    Dina Demner-Fushman, Marc D. Kohli, Marc B. Rosenman, Sonya E. Shooshan, Laritza
    Rodriguez, Sameer Antani, George R. Thoma, and Clement J. McDonald. Preparing a
    collection of radiology examinations for distribution and retrieval. Journal of the American
    Medical Informatics Association, 2016. doi: 10.1093/jamia/ocv080.

    Views have been determined by projection using T-SNE.  To use the T-SNE view rather than the
    view defined by the record, set use_tsne_derived_view to true.

    Dataset website:
    https://openi.nlm.nih.gov/faq

    Download images:
    https://academictorrents.com/details/5a3a439df24931f410fac269b87b050203d9467d
    """

    def __init__(
        self,
        root_path,
        args=None,
        transforms=None,
    ):
        super(Openi_Dataset, self).__init__()

        # np.random.seed(seed)  # Reset the seed so all runs are the same.
        # self.imgpath = imgpath
        # self.transform = transform
        # self.data_aug = data_aug

        self.root_path = root_path
        self.pathologies = [
            # NIH
            "Atelectasis",
            "Cardiomegaly",
            "Effusion",
            "Infiltration",
            "Mass",
            "Nodule",
            "Pneumonia",
            "Pneumothorax",
            ## "Consolidation",
            "Edema",
            "Emphysema",
            "Fibrosis",
            "Pleural_Thickening",
            "Hernia",
            # ---------
            "Fracture",
            "Opacity",
            "Lesion",
            # ---------
            "Calcified Granuloma",
            "Granuloma",
            # ---------
            "No_Finding",
        ]
        # self.pathologies = sorted(self.pathologies)
        # self.pathologies.append("No_Finding")

        mapping = dict()
        mapping["Pleural_Thickening"] = ["pleural thickening"]
        mapping["Infiltration"] = ["Infiltrate"]
        mapping["Atelectasis"] = ["Atelectases"]

        # Load data
        self.imgpath = os.path.join(self.root_path, "NLMCXR_png")
        self.xmlpath = os.path.join(self.root_path, "NLMCXR_reports")
        self.csv_path = os.path.join(self.root_path, "custom.csv")
        self.csv = pd.read_csv(self.csv_path)
        self.csv = self.csv.replace(np.nan, "-1")
        self.transform = transforms

        self.gt = []
        for pathology in self.pathologies:
            mask = self.csv["labels_automatic"].str.contains(pathology.lower())
            if pathology in mapping:
                for syn in mapping[pathology]:
                    # print("mapping", syn)
                    mask |= self.csv["labels_automatic"].str.contains(syn.lower())
            self.gt.append(mask.values)

        self.gt = np.asarray(self.gt).T
        self.gt = self.gt.astype(np.float32)
        # Rename pathologies
        self.pathologies = np.char.replace(self.pathologies, "Opacity", "Lung Opacity")
        self.pathologies = np.char.replace(self.pathologies, "Lesion", "Lung Lesion")

        self.gt[np.where(np.sum(self.gt, axis=1) == 0), -1] = 1

        # for i in range(len(self.pathologies)):
        #     logger.bind(stage="DATA").info(
        #         f"{self.pathologies[i]} -> {len(np.where(self.gt[:,i]==1.0)[0])}"
        #     )
        org_data_len = self.gt.shape[0]
        if args.trim_data:
            self.trim(args.train_data)
            # Assign no finding
            row_sum = np.sum(self.gt, axis=1)
            # self.gt[np.where(row_sum == 0), -1] = 1
            drop_idx = np.where(row_sum == 0)[0]
            if len(drop_idx) != 0:
                self.gt = np.delete(self.gt, drop_idx, axis=0)
                # self.imgs = np.delete(self.imgs, drop_idx, axis=0)
                self.csv = self.csv.drop(self.csv.iloc(drop_idx).index)

        # logger.bind(stage="DATA").info(f"Num of no_finding: {(self.gt[:,-1]==1).sum()}")
        logger.bind(stage="DATA").info(
            f"Trimed data size: {org_data_len-self.gt.shape[0]}"
        )
        logger.bind(stage="DATA").info(
            f"Maximum labels for an individual image: {np.sum(self.gt, axis=1).max()}"
        )

    def trim(self, train_data):
        MIMIC_CUT_LIST = [
            "Infiltration",
            "Mass",
            "Nodule",
            "Emphysema",
            "Fibrosis",
            "Pleural_Thickening",
            "Hernia",
            "Calcified Granuloma",
            "Granuloma",
        ]

        # Cut label
        NIH_CUT_LIST = [
            "Calcified Granuloma",
            "Granuloma",
            "Lung Lesion",
            "Lung Opacity",
            "Fracture",
        ]
        if train_data == "MIMIC":
            cut_list = MIMIC_CUT_LIST
        elif train_data == "NIH":
            cut_list = NIH_CUT_LIST
        else:
            raise ValueError(f"TRAIN DATA {train_data}")

        for dp_class in cut_list:
            drop_idx_col = np.where(self.pathologies == dp_class)[0].item()
            drop_idx_row = np.where(self.gt[:, drop_idx_col] == 1.0)[0]
            if len(drop_idx_row) == 0:
                print(f"skip {dp_class}")
                continue
            self.pathologies = np.delete(self.pathologies, drop_idx_col)
            self.gt = np.delete(self.gt, drop_idx_row, axis=0)
            self.gt = np.delete(self.gt, drop_idx_col, axis=1)
            self.csv = self.csv.drop(self.csv.iloc[drop_idx_row].index)
        return

    def string(self):
        return self.__class__.__name__ + " num_samples={}".format(len(self))

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        sample = {}
        file_name = self.csv.iloc[idx].file_name
        image = Image.fromarray(
            io.imread(os.path.join(self.imgpath, file_name))
        ).convert("RGB")
        image = self.transform(image)

        label = self.gt[idx]
        return image, label, idx


def construct_openi_cut(args, root_dir, mode):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if mode == "train":
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
                transforms.Resize((args.resize, args.resize)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    dataset = Openi_Dataset(root_path=root_dir, transforms=transform, args=args)
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True if mode == "train" else False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )

    return loader

    # dataset = Openi_Dataset(transform)


# construct_loader(args, "")
