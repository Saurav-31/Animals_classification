import matplotlib.pyplot as plt
from torchvision import utils
import torch
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

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


def eval_results_epoch(pred_list, gt_list, classes):
    pred = np.concatenate(pred_list, 0)
    gt = np.concatenate(gt_list, 0)
    print(confusion_matrix(gt, pred))
    cm = pd.DataFrame(confusion_matrix(gt, pred), columns=classes)
    cm.to_csv("Confusion_matrix.csv", index=classes.all())

    fn = cm.apply(lambda x: x.sum(), axis=1) - np.diag(cm)
    fp = cm.apply(lambda x: x.sum(), axis=0) - np.diag(cm)
    tp = np.diag(cm)
    eval = pd.DataFrame({'Classes': classes, 'TP': tp, 'FP': fp.values, 'FN': fn.values})
    print(eval)


def view_predictions(data_iter, device, model_ft, animals_dataset):
    plt.figure(figsize=(10, 10))
    for i in range(12):
        sample = next(data_iter)
        data, labels = sample['image'], sample['labels']

        img = data
        img = torch.unsqueeze(img, 0)
        img = img.float().to(device)

        out = model_ft(img)
        prediction = torch.nn.functional.softmax(out)
        predicted_label = animals_dataset.labels_to_idx[torch.nn.functional.softmax(out).argmax().item()]
        print("Predicted Label:", predicted_label)
        print("Probability:", torch.nn.functional.softmax(out).max().item())
        img = torch.squeeze(img).cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        plt.subplot(4, 3, i + 1)
        plt.imshow(img)
        plt.title(predicted_label)
        plt.text(240, 240, torch.nn.functional.softmax(out).max().item())
        plt.axis('off')
    plt.show()

def evaluate_testset(model_ft, test_loader, device):
    soft_out = []
    for i, sample in enumerate(test_loader):
        data, labels = sample['image'].float().to(device), sample['labels'].float().to(device)
        out = model_ft(data)
        print(i, out.size())
        soft_out.append(torch.nn.functional.softmax(out).cpu().detach().numpy())

    print(len(soft_out))
    print(type(soft_out[0]))
    test_out = np.concatenate(soft_out, 0)
    test_out.shape

    return test_out

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
