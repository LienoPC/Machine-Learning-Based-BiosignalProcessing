import torch
import torch.utils.data.dataset as dataset
import cv2
from PIL import Image
import torchvision.transforms as transforms


class SignalImageDataset(dataset.Dataset):
    """
    Dataset that elaborates scalogram images

    img_list: list of images path
    label_list: path of the list of labels
    """

    def __init__(self, img_list, label_list, transform=None):
        self.img_list = img_list
        self.label_list = label_list
        # Transform operations to be applied on input images
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        img_path = self.img_list[item]
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        label = self.label_list[item]

        return image, label

    @staticmethod
    def compute_mean_std(dataloader, size):
        psum = torch.tensor([0.0])  # Pixel sum
        psum_sq = torch.tensor([0.0])  # Squared sum

        for image in dataloader:
            psum += image.sum(axis=[0, 2, 3])
            psum_sq += (image ** 2).sum(axis=[0, 2, 3])

        # Pixel count
        count = len(dataloader) * size

        total_mean = psum / count
        total_var = (psum_sq / count) - (total_mean ** 2)
        total_std = torch.sqrt(total_var)
        print('- mean: {:.4f}'.format(total_mean.item()))
        print('- std:  {:.4f}'.format(total_std.item()))
        return total_mean, total_std


class ScalogramImageTransform():

    def __init__(self, resize_dim, mean, std):
        self.transform = transforms.Compose([
            transforms.Resize(resize_dim),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def transform(self, data):
        return self.transform(data)


