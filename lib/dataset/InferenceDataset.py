# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bowen Cheng (bcheng9@illinois.edu) and Bin Xiao (leoxiaobin@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from collections import OrderedDict
import logging
import os
import os.path
import glob

import cv2
import json_tricks as json
import numpy as np
from torch.utils.data import Dataset

from utils import zipreader

logger = logging.getLogger(__name__)


class InferenceDataset(Dataset):
    """`CrowdPose`_ Dataset.

    Args:
        root (string): Root directory where dataset is located to.
        dataset (string): Dataset name(train2017, val2017, test2017).
        data_format(string): Data format for reading('jpg', 'zip')
        transform (callable, optional): A function/transform that  takes in an opencv image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, transform=None):
        self.name = 'CUSTOM'
        self.root = root
        self.images = glob.glob(f"{root}/*")
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        file_name = self.images[index]

        img = cv2.imread(
            file_name,
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.images)
