import collections
import os
from pickletools import read_unicodestring1
import numpy as np
import pandas as pd
import skimage.transform
from skimage import io
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from multiprocessing import Pool, cpu_count
import torchvision.transforms.functional as TF
from tqdm import tqdm

root_dir = "/mnt/HD/Dataset/PADCHEST"
# files = os.listdir("/mnt/HD/Dataset/PADCHEST/pc_test_images")

org = pd.read_csv("/mnt/HD/Dataset/PADCHEST/DATA/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv", dtype=str)

org = org[org.MethodProjection == "Manual review of DICOM fields"]

org = org[org.MethodLabel == "Physician"]

org = org.loc[(org.Projection == "PA") | (org.Projection == "AP")]

org.to_csv("/mnt/HD/Dataset/PADCHEST/PADCHEST_CLEARN_PA_AP.csv")

df = pd.read_csv("/mnt/HD/Dataset/PADCHEST/PADCHEST_TEST_CLEAN_PA_AP.csv", dtype=str)

files = df.ImageID.to_list()


def tmpFunc(img_path):
    img = Image.open(img_path)
    table = [i / 256 for i in range(65536)]
    img = img.point(table, "L")
    img = TF.resize(img, (512, 512))
    name = img_path.split("/")[-1]
    img.save(os.path.join(root_dir, "PADCHEST_TEST_CLEAN_PA_AP/{}".format(name)))
    return


def applyParallel(func):
    with Pool(cpu_count() - 12) as p:
        p.map(
            func,
            [
                os.path.join(root_dir, "images", files[i])
                for i in tqdm(range(len(files)))
            ],
        )


if __name__ == "__main__":
    processed_df = applyParallel(tmpFunc)
