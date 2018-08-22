import matplotlib.pyplot as plt
from torchvision import utils
import torch

# Helper function to show a batch
def show_keypoints_batch(sample_batched):
    """Show image with keypoints for a batch of samples."""
    images_batch, keypoints_batch = sample_batched['image'], sample_batched['keypoints'].reshape(-1, 14, 2)
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(keypoints_batch[i, :, 0].deatch().numpy() + i * im_size,
                    keypoints_batch[i, :, 1].detach().numpy(),
                    s=10, marker='.', c='r')
        plt.title('Batch from dataloader')

