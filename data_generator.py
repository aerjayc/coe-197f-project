import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import PIL.Image
import os.path


class SemanticSegmentationDataset(Dataset):
    def __init__(self, data_dir, gt_fname, cuda=True):
        self.data_dir = data_dir
        gt_path = os.path.join(data_dir, gt_fname)

        self.gt_dict = np.load(gt_path,
                               allow_pickle=True).flat[0]
        self.img_names = list(self.gt_dict.keys())

        self.cuda = cuda

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.img_names[idx]
        img_path = os.path.join(self.data_dir, img_name)
        img = PIL.Image.open(img_path)
        img = transforms.ToTensor()(img)    # C, H, W

        gt = self.gt_dict[img_name].transpose(2,0,1)    # C, H, W
        gt = torch.from_numpy(gt).long()

        if cuda:
            img = img.cuda()
            gt = gt.cuda()

        return img, gt  # torch (cuda) CHW
    
    def get_img_names(self, sorted=True):
        """Returns the list of image files in `self.data_dir`,
            but only in that directory (no recursive search)
        """
        img_exts = {'.jpg', '.jpeg', '.png', '.bmp'}

        img_names = []
        _, _, files = next(os.walk(self.img_names))
        for fname in sorted(files):
            if os.path.splitext(fname)[1].lower() in img_exts:
                img_names.append(fname)
        
        return img_names

