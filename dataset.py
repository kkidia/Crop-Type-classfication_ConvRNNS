# ================DataSet================


# @Lib:
import os
import time
import numpy as np
import torch
import gcsfs
from torch.utils.data import Dataset
from typing import Dict
import warnings

warnings.filterwarnings('ignore')


# Month mapping helper

MAPS = {
    'class': {0:0, 1:1, 2:2},
    'sub': {0:0, 1:1, 2:2, 3:3, 5:4, 6:5, 7:6},
    'name': {
        0:0,2:1,3:2,4:3,6:4,7:5,10:6,11:7,12:8,13:9,14:10,15:11,16:12,18:13,
        19:14,21:15,22:16,23:17,26:18,27:19,28:20,29:21,30:22,31:23,33:24,
        34:25,35:26,36:27,37:28,38:29,39:30,41:31,42:32,45:33,46:34,47:35,
        49:36,50:37,51:38,53:39,54:40,55:41,56:42,61:43,62:44,63:45,64:46
    },
    'year': {0:0, 2020:1, 2023:2},
    'month': {8.0:0, 9.0:1, 10.0:2}
}
NUM_CLASSES = {k: len(v) for k, v in MAPS.items()}

# Month mapping helper
MONTH_MAP = {
    "Jan":1, "Feb":2, "Mar":3, "Apr":4, "May":5, "Jun":6,
    "Jul":7, "Aug":8, "Sep":9, "Oct":10, "Nov":11, "Dec":12
}


def map_array_to_idx(arr, value_to_idx):
    """
    Map each value in a numpy array to its index using the provided dictionary.
    Handles any value not in value_to_idx by defaulting to 0.
    """
    return np.vectorize(lambda v: value_to_idx.get(v, 0))(arr).astype(np.int64)

class NumpyDataset(torch.utils.data.Dataset):
    """Multi-temporal dataset reading .npy patches from GCS, with true label shapes and metadata."""

    def __init__(self, root_dir, time_steps=3, strategy="sliding",
                 max_windows=5, project_id=None, credentials_file=None):
        self.time_steps = time_steps
        self.strategy = strategy
        self.max_windows = max_windows

        bucket = root_dir[5:] if root_dir.startswith("gs://") else root_dir
        self.fs = gcsfs.GCSFileSystem(project=project_id, token=credentials_file) \
                  if project_id else gcsfs.GCSFileSystem()

        # Get all .npy file paths
        self.file_paths = self.fs.glob(os.path.join(bucket, "*", "*.npy"))
        print(f"Found {len(self.file_paths)} .npy files")

        # Group by plot ID
        self.samples = self._create_samples()
        print(f"Created {len(self.samples)} samples using {self.strategy} strategy")

    def _create_samples(self):
        groups = {}
        for path in self.file_paths:
            basename = os.path.basename(path)
            parts = basename.split("_")
            if len(parts) < 2:
                continue
            plot_id = parts[1]  
            groups.setdefault(plot_id, []).append(path)

        samples = []
        for plot_id, files in groups.items():
            # Sort by month from parent folder name
            files = sorted(files, key=lambda p: MONTH_MAP.get(
                os.path.basename(os.path.dirname(p)), 0))

            if self.strategy == "original":
                if len(files) >= self.time_steps:
                    samples.append(files[:self.time_steps])
            else:  # sliding window
                n_windows = min(len(files) - self.time_steps + 1, self.max_windows)
                for i in range(max(0, n_windows)):
                    samples.append(files[i:i + self.time_steps])

        return samples

    def _load_npy_from_gcs(self, path):
        with self.fs.open(path, 'rb') as f:
            return np.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        files = self.samples[idx]
        images, months, years = [], [], []

        for t, file in enumerate(files):
            data = self._load_npy_from_gcs(file)  # shape: [9, H, W]
            img = data[0:4]           # [4, H, W]
            class_label = data[4]     # [H, W]
            sub_class_label = data[5]
            name_label = data[6]
            year_val = int(data[7, 0, 0])   # [scalar], constant across patch
            month_val = float(data[8, 0, 0])# [scalar], constant across patch

            images.append(img)
            months.append(month_val)
            years.append(year_val)

            if t == 0:
                class_label_idx = map_array_to_idx(class_label, MAPS['class'])
                sub_class_label_idx = map_array_to_idx(sub_class_label, MAPS['sub'])
                name_label_idx = map_array_to_idx(name_label, MAPS['name'])
                year_idx = MAPS['year'].get(year_val, 0)
                # Map month to index for meta
                month_idx = MAPS['month'].get(month_val, 0)

        image = torch.from_numpy(np.stack(images, 0)).float()   

        return {
            'image': image,   
            'label': {
                'class_label': torch.from_numpy(class_label_idx).long(),         
                'sub_label': torch.from_numpy(sub_class_label_idx).long(),       
                'name_label': torch.from_numpy(name_label_idx).long(),           
                'year': torch.tensor([year_idx], dtype=torch.long)               
            },
            'meta': torch.tensor([month_idx], dtype=torch.long),                 
            'all_months': torch.tensor(months, dtype=torch.float),               
            'all_years': torch.tensor(years, dtype=torch.long)                   
        }
