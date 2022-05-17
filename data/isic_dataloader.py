from torch.utils.data import Dataset, DataLoader
import numpy as np
from skimage import io
import torchvision.transforms as transforms
import pickle
import torch
import os
import pandas as pd
import random
from PIL import Image

Labels = {
    "MEL": 0,
    "NV": 1,
    "BCC": 2,
    "AK": 3,
    "BKL": 4,
    "DF": 5,
    "VASC": 6,
    "SCC": 7,
    "UNK": 8,
}


class ISICTrain(Dataset):
    def __init__(self, data, noise_targets, clean_targets, transform) -> None:
        self.data = data
        self.noise_targets = noise_targets
        self.clean_targets = clean_targets
        self.transform = transform

    def __getitem__(self, index):
        img_path, clean_label, noise_label = (
            self.data[index],
            self.clean_targets[index],
            self.noise_targets[index],
        )
        img = Image.fromarray(
            io.imread(
                os.path.join(
                    "/run/media/Data/ISIC2019/ISIC_2019_Training_Input",
                    img_path + ".jpg",
                )
            )
        ).convert("RGB")
        img_t = self.transform(img)
        return img_t, noise_label, clean_label, index

    def __len__(self):
        return self.data.shape[0]


class ISICDataset(Dataset):
    def __init__(
        self, root_dir, transforms=None, mode=None, args=None, num_noise=0.4
    ) -> None:
        self.transform = transforms
        self.root_dir = root_dir
        self.mode = mode
        df_path = os.path.join(root_dir, "ISIC_2019_Training_GroundTruth.csv")
        df = pd.read_csv(df_path)
        img_list = df.iloc[:, 0].values
        gt = df.iloc[:, 1:-1].values.astype(int)
        # one_hot = torch.argmax(torch.from_numpy(gt), dim=1)
        total_targets = torch.from_numpy(gt).permute(1, 0)
        # total_idx = list(range(one_hot.shape[0]))
        # sample 628 for each class
        self.imgs = []
        self.clean_targets = []
        for i in range(8):
            if i == 5 or i == 6:
                target_num = total_targets[i].nonzero().shape[0]
            else:
                target_num = 628
            class_idx = total_targets[i].nonzero()
            total_idx = list(range(total_targets[i].nonzero().shape[0]))
            random.shuffle(total_idx)
            self.imgs.extend(
                img_list[class_idx[total_idx[:target_num]]].squeeze().tolist()
            )
            self.clean_targets.extend((torch.zeros(target_num).int() + i).tolist())

        total_idx = list(range(len(self.clean_targets)))
        random.shuffle(total_idx)
        num_train = int(0.8 * len(self.clean_targets))
        train_idx = total_idx[:num_train]
        test_idx = total_idx[num_train:]
        self.train_imgs = np.array(self.imgs)[train_idx]
        self.test_imgs = np.array(self.imgs)[test_idx]

        self.train_labels = np.array(self.clean_targets)[train_idx].tolist()
        self.test_labels = np.array(self.clean_targets)[test_idx].tolist()

        # inject noise
        noise_label = []
        total_idx = list(range(len(self.train_labels)))
        random.shuffle(total_idx)
        num_noise = int(num_noise * len(self.train_labels))
        noise_idx = total_idx[:num_noise]
        for i in range(len(self.train_labels)):
            if i in noise_idx:
                noiselabel = random.randint(0, 7)
                noise_label.append(noiselabel)
            else:
                noise_label.append(self.train_labels[i])
        self.noisy_labels = np.array(noise_label)

        # Train:Test = 80:20
        # total_idx = list(range(len(self.clean_targets)))
        # random.shuffle(total_idx)
        # num_train = int(0.8 * len(self.clean_targets))
        # train_idx = total_idx[:num_train]
        # test_idx = total_idx[num_train:]
        # self.train_imgs = np.array(self.imgs)[train_idx]
        # self.test_imgs = np.array(self.imgs)[test_idx]

        # self.train_labels = np.array(noise_label)[train_idx]
        # self.test_labels = np.array(noise_label)[test_idx]

    def calc_prior(self):
        _, count = np.unique(self.noisy_labels, return_counts=True)
        return count / self.noisy_labels.shape[0]


def construct_isic(root_dir):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    dataset = ISICDataset(root_dir)
    train_dataset = ISICTrain(
        dataset.train_imgs,
        dataset.train_labels,
        dataset.noisy_labels,
        transform=train_transform,
    )
    eval_train_dataset = ISICTrain(
        dataset.train_imgs,
        dataset.train_labels,
        dataset.noisy_labels,
        transform=test_transform,
    )
    test_dataset = ISICTrain(
        dataset.test_imgs,
        dataset.test_labels,
        dataset.test_labels,
        transform=test_transform,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )
    eval_train_loader = DataLoader(
        dataset=eval_train_dataset,
        batch_size=64,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=64,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, test_loader, eval_train_loader, dataset.calc_prior()


if __name__ == "__main__":
    ISICDataset("/run/media/Data/ISIC2019/")
