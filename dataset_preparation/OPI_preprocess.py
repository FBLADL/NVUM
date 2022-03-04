import pandas as pd
import csv
import numpy as np
import os
import shutil
from tqdm import tqdm
import tarfile
import xml
import xml.etree.ElementTree as ET
from glob import glob

path = "/mnt/HD/Dataset/open-i"

view_df = pd.read_csv(os.path.join(path, "indiana_projections.csv"), dtype=str)

ft_df = view_df[view_df.projection == "Frontal"]

id_format = "CXR{}"


for i in tqdm(range(len(ft_df))):
    a = 1
    curr = ft_df.iloc[i]
    target_name = f"CXR{curr.filename[:-8]}.png"
    os.path.join(path, "NLMCXR_png")
    shutil.copy(
        os.path.join(path, "NLMCXR_png", target_name), os.path.join(path, "NLMCXR_Frontal_png")
    )

xml_path = "/mnt/HD/Dataset/open-i/NLMCXR_reports/ecgen-radiology"
png_path = "/mnt/HD/Dataset/open-i/NLMCXR_FRONTAL_png"

columns = ["uid", "file_name", "labels_major", "labels_automatic"]

df = pd.DataFrame(columns=columns)

files = os.listdir(xml_path)
files.sort(key=lambda f: int(f[:-4]))

for filename in files:
    tree = ET.parse(os.path.join(xml_path, filename))
    root = tree.getroot()
    uid = root.find("uId").attrib["id"]
    labels_m = [node.text.lower() for node in root.findall(".//MeSH/major")]
    labels_m = "|".join(np.unique(labels_m))
    labels_a = [node.text.lower() for node in root.findall(".//MeSH/automatic")]
    labels_a = "|".join(np.unique(labels_a))
    image_nodes = root.findall(".//parentImage")

    file_name = glob(os.path.join(png_path, f"{uid}_*"))
    if len(file_name) > 1 or len(file_name) == 0:
        print(f"{len(file_name)} --- {file_name}")
    # if len(file_name) ==0:
    #     print(f"{len(file_name)} --- {file_name}")

    for item in file_name:
        file_name = file_name[0]
        tmp = pd.DataFrame(
            {
                "uid": [uid],
                "file_name": [item.split("/")[-1]],
                "labels_major": [labels_m],
                "labels_automatic": [labels_a],
            }
        )
        df = df.append(tmp)

a = 1

# df.to_csv("/mnt/HD/Dataset/open-i/custom.csv")
