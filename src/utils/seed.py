from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed common RNGs used by the project.

    Torch is imported lazily so preprocessing can run in minimal environments
    that only install pandas/pyarrow/PyYAML.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    except ImportError:
        return
