from skimage.transform import rotate, resize
import numpy as np
import torch
import torchvision.transforms.functional as f
import numbers
class Rotate2d(object):
    def __init__(self):
        import random
        self.degree = random.randint(-30, 30)

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']
        image = rotate(image, self.degree)
        dim = image.shape[1]
        #labels = self.rot_pts(labels, self.degree * np.pi / 180, dim)
        labels = labels
        return {'image': image, 'labels': labels}

    @staticmethod
    def rot_pts(pts, theta, dim):
        flip_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        l = np.array([[i[0] - dim/2, -i[1] + dim/2] for i in pts.reshape(14, 2)])
        p = [np.matmul(flip_matrix, l[i]) for i in range(14)]
        q = [[i[0] + dim/2, -i[1] + dim/2] for i in p]
        q = np.array(q).reshape(-1)
        return q


class flip2d(object):
    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']
        image, labels = self.flip_img(image, labels)
        return {'image': image, 'labels': labels}

    @staticmethod
    def flip_img(img, points):
        new_img = np.flip(img, 1)
        # new_img = rotate(img, -180)
        return new_img.copy(), np.array([[100 - i[0], i[1]] for i in points.reshape(-1, 2)]).reshape(-1)


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = resize(image, (new_h, new_w))

        # h and w are swapped for labels because for images,
        # x and y axes are axis 1 and 0 respectively
        labels = labels * [new_w / w, new_h / h]

        return {'image': img, 'labels': labels}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'labels': labels}


class ToTensor(object):

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']
        # print((image > 0).sum())
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'labels': labels}


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']

        return {'image': f.normalize(image, self.mean, self.std),
                'labels': labels}

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class CenterCrop(object):
    """Crops the given PIL Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        image, labels = sample['image'], sample['labels']
        print(image.size)
        return {'image': f.center_crop(image, self.size), 'labels': labels}
