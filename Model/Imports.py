import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import make_grid

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.utils import save_image, make_grid

from scipy.signal import savgol_filter

from six.moves import xrange


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


