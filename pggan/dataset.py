from pathlib import Path

import torch
from torch.utils.data import Dataset


class CelebAHQ(Dataset):
    def __init__(
        self,
        data_root:  Path,
        split:      str,
    ) -> None:
        super(CelebAHQ, self).__init__()
        data_paths = data_root.joinpath(split)

    def __getitem__(
        self,
        idx:    int
    ) -> torch.Tensor:
        return idx

    def __len__(self):
        return 0
