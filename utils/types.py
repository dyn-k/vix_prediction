from typing import *

import pandas as pd

from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

# pandas types
DataFrame = NewType("DataFrame", pd.DataFrame)
Series = NewType("Series", pd.Series)

# torch types
TorchDataset = NewType("torchDataset", Dataset)
TorchDataLoader = NewType("torchDataLoader", DataLoader)
TorchTensor = NewType("torchTensor", Tensor)

# pytorch-lightning types
LightningModule = NewType("LightningModule", pl.LightningModule)