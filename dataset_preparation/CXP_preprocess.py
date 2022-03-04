import pandas as pd
import random
import numpy as np
from numpy.core.defchararray import add

df = pd.read_csv("/run/media/Data/CheXpert-v1.0-small/train.csv", index_col=0)

# filter Lateral
df = df[df["Frontal"] == "Frontal"]
# filter age 0
df = df[df.Age != 0]
# fill NAN with 0
df = df.fillna(0.0)
# fill -1 with 0
df = df.replace(-1.0, 0.0)

index = df.index.to_numpy()
patients = [i.split("/")[2] for i in index]
df["patients"] = patients
random.shuffle(patients)
train_num = int(len(patients) * 0.9)
val_num = len(patients) - train_num
train_split = patients[:train_num]
val_split = patients[train_num : val_num + train_num]

new_df = df.reset_index().set_index("patients")

train_df = new_df.loc[pd.Index(train_split)]
val_df = new_df.loc[pd.Index(val_split)]

pop = train_df.pop("No Finding")
train_df = pd.concat([train_df, pop], 1)
pop = val_df.pop("No Finding")
val_df = pd.concat([val_df, pop], 1)


train_df.to_csv("/run/media/Data/CheXpert-v1.0-small/filtered_train.csv")
val_df.to_csv("/run/media/Data/CheXpert-v1.0-small/filtered_test.csv")