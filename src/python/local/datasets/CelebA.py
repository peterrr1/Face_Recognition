from torch.utils.data import Dataset
import torch
from typing import Union, Optional, Callable, Tuple, Any
import pandas as pd
import os
from PIL import Image
from pathlib import Path

class CelebA(Dataset):
    def __init__(
            self,
            root: Union[str, Path],
            transform: Optional[Callable] = None,
            
    ):
        self.root = root
        self.transform = transform
        
        # Read filenames and attributes
        files, attr = self._read_csv('list_attr_celeba.csv')
        
        self.attr = torch.div(attr + 1, 2, rounding_mode='floor').float()
        self.files = [os.path.join(self.root, 'img_align_celeba/', file) for file in files]
        

    def get_pos_weights(self) -> torch.Tensor:
        num_of_labels = len(self.attr)
        num_of_pos_labels = torch.sum(self.attr, dim = 0)
        num_of_neg_labels = num_of_labels - num_of_pos_labels
        pos_weights = num_of_neg_labels / num_of_pos_labels
        return num_of_labels, pos_weights


    def _read_csv(
            self,
            filename: str,
    ) -> Union[torch.Tensor, Tuple]:
        
        df = pd.read_csv(os.path.join(self.root, filename), index_col=0, header=0)
        attr = torch.from_numpy(df.values)
        files = df.index.values

        return files, attr
    

    def __getitem__(self, index) -> Tuple[Any, Any]:
        image = Image.open(self.files[index])
        
        if self.transform is not None:
            image = self.transform(image)

        return image, self.attr[index]


    def __len__(self) -> int:
        return len(self.attr)

    
