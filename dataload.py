from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
from sklearn.preprocessing import LabelEncoder, LabelBinarizer

class AnimalsDataset(Dataset):

    def __init__(self, filename, root_dir, transform=None, train=False):
        df = pd.read_csv(filename)
        le = LabelEncoder()
        lb = LabelBinarizer()
        self.train = train
        if self.train:
            df['Animal'] = le.fit_transform(df.Animal)
            self.labels_one_hot = lb.fit_transform(df.Animal)
            self.labels_to_idx = le.classes_
        else:
            df['Animal'] = 0
        self.fk_frame = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.fk_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.fk_frame.iloc[idx, 0])
        image = imread(img_name)
        if self.train:
            labels = self.labels_one_hot[idx]
        else:
            labels = 0
        #image = rgb2gray(image)
        image = resize(image, (256, 256, 3))

        if image is not None:
            sample = {'image': image, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)

        return sample

