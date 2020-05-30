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

if __name__ == '__main__':
    data_dir = 'drinks/'
    gt_fname = 'segmentation_train.npy'
    images, labels = SemanticSegmentationDataset(data_dir, gt_fname, cuda=False)
    images, labels = images.permute(1,2,0), labels.permute(1,2,0)

    import matplotlib.pyplot as plt

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Input image', fontsize=14)
    plt.imshow(images[0])
    plt.savefig("input_image.png", bbox_inches='tight')
    plt.show()

    labels = labels * 255
    masks = labels[..., 1:]
    bgs = labels[..., 0]

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Semantic segmentation', fontsize=14)
    plt.imshow(masks[0])
    plt.savefig("segmentation.png", bbox_inches='tight')
    plt.show()

    shape = (bgs[0].shape[0], bgs[0].shape[1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Background', fontsize=14)
    plt.imshow(np.reshape(bgs[0], shape), cmap='gray', vmin=0, vmax=255)
    plt.savefig("background.png", bbox_inches='tight')
    plt.show()
