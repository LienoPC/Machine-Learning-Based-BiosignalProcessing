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
    def compute_mean_std(dataloader):
        psum = torch.tensor([0.0, 0.0, 0.0])
        psum_sq = torch.tensor([0.0, 0.0, 0.0])
        n_pixels = 0

        for images, _ in dataloader:
            # images: (B, C, H, W)
            B, C, H, W = images.shape
            n_pixels += B * H * W

            psum += images.sum(dim=[0, 2, 3])
            psum_sq += (images ** 2).sum(dim=[0, 2, 3])

        total_mean = psum / n_pixels
        total_var = (psum_sq / n_pixels) - total_mean ** 2
        total_std = torch.sqrt(total_var)

        print(f"- mean: {total_mean}")
        print(f"- std:  {total_std}")
        return total_mean, total_std


class ScalogramImageTransform():

    def __init__(self, resize_dim, mean=None, std=None):
        transform_list = [
            transforms.Resize(resize_dim),
            transforms.ToTensor()
        ]

        if mean is not None and std is not None:
            transform_list.append(transforms.Normalize(mean, std))

        self.transform = transforms.Compose(transform_list)


    def get_transform(self):
        return self.transform


