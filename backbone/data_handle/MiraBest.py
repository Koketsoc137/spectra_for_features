import os
import pickle

import numpy as np
import torchvision as tv
from PIL import Image
from torch.utils.data import Dataset


class MiraBest(Dataset):
    """MiraBest FRI (0) / FRII (1) from the MiraBest_F pickled batches.

    subset="full" keeps [100, 102, 104, 110, 112] / [200, 201, 210] (MBFRFull),
    subset="confident" keeps [100, 102, 104] / [200, 201] (MBFRConfident).
    Returns (image, label, name) like Custom.Custom_labelled.
    """

    def __init__(self, root, train=True, subset="full", resize=224, crop=224,
                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        files = [f"data_batch_{i}" for i in range(1, 8)] if train else ["test_batch"]
        data, targets, names = [], [], []
        for file_name in files:
            with open(os.path.join(root, file_name), "rb") as f:
                entry = pickle.load(f, encoding="latin1")
            data.append(entry["data"])
            targets.extend(entry["labels"])
            names.extend(entry["filenames"])

        # label indices into batches.meta label_names ['100','102','104','110','112','200','201','210','300','310']
        fr1, fr2 = ([0, 1, 2, 3, 4], [5, 6, 7]) if subset == "full" else ([0, 1, 2], [5, 6])
        targets = np.array(targets)
        keep = np.isin(targets, fr1 + fr2)
        self.data = np.vstack(data).reshape(-1, 150, 150)[keep]
        self.targets = np.isin(targets[keep], fr2).astype(np.int64).tolist()
        self.names = np.array(names)[keep].tolist()

        self.transform = tv.transforms.Compose([
            tv.transforms.Resize(resize),
            tv.transforms.CenterCrop(crop),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=mean, std=std),
        ])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        # grey -> RGB, matching how ImageFolder loads the RGZ PNGs
        image = Image.fromarray(self.data[index], mode="L").convert("RGB")
        return self.transform(image), self.targets[index], self.names[index]
