from pathlib import Path

import cv2 as cv
import torch
from torch.utils.data import Dataset


class CelebAHQ(Dataset):
    def __init__(
        self,
        data_root:  Path,
        split:      str,
    ) -> None:
        super(CelebAHQ, self).__init__()
        data_paths = data_root.joinpath(split).glob('*.jpg')
        self.im_list = [im for im in data_paths]

    def __getitem__(
        self,
        idx:    int
    ) -> torch.Tensor:
        im_np = cv.imread(str(self.im_list[idx]))[
            :, :, [2, 1, 0]]  # BGR to RGB
        im_tensor = torch.FloatTensor(
            im_np/255).permute(2, 0, 1)  # [0, 255] to [0, 1]
        return im_tensor

    def __len__(self):
        return len(self.im_list)
