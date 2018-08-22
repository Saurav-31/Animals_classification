from torch.utils.data import Dataset
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from PIL import Image
from torchvision import transforms


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
        image = Image.open(img_name)
        image = image.convert('RGB')
        jpg_to_tensor = transforms.ToTensor()
        tensor_to_pil = transforms.ToPILImage()
        image = tensor_to_pil(jpg_to_tensor(image))
        if self.train:
            #labels = self.labels_one_hot[idx]
            labels = self.fk_frame.iloc[idx, 1]
        else:
            labels = 0

        if image is not None:
            sample = {'image': image, 'labels': labels}

        if self.transform:
            image = self.transform(image)
            sample = {'image': image, 'labels': labels}

        return sample
