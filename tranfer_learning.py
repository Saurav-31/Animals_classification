from __future__ import print_function, division
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from cnn_model import Net, train, test, class_weights
from dataload import *
from data_transform import *
from data_utils import *
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.transforms.functional
from torchsample.transforms import RandomAffine, Rotate, Zoom
from data_loader_pil import *
import time
import warnings
warnings.filterwarnings(action='ignore')
plt.ion()

size = 224
# Data augmentation and normalization for training
# Just normalization for validation
## Activate it for normal data loader
# data_transform_train = transforms.Compose([
#         Rotate2d(),
#         RandomCrop((size, size)),
#         ToTensor(),
#         Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
# data_transform_test = transforms.Compose([
#         RandomCrop((size, size)),
#         ToTensor(),
#         Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])

data_transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(size),
        transforms.RandomAffine(30, translate=[0, 0.2], scale=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # RandomAffine(rotation_range=30, translation_range=0.2, zoom_range=(0.75, 1.25)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

data_transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

data_transform = transforms.Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
data_transform1 = transforms.Compose([ToTensor()])

animals_dataset = AnimalsDataset(filename='meta-data_new/meta-data/train.csv', root_dir='./train_new/train/', train=True,
                                 transform=data_transform_train)

test_dataset = AnimalsDataset(filename='meta-data_new/meta-data/test.csv', root_dir='./test_new/test', train=False,
                              transform=data_transform_test)
data_iter = iter(animals_dataset)
sample = next(data_iter)
data, labels = sample['image'], sample['labels']
print(data.max(), data.min())

validation_split = 0.1
shuffle_dataset = True
random_seed = 42

# Creating data indices for training and validation splits:
dataset_size = len(animals_dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(animals_dataset, batch_size=32, sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(animals_dataset, batch_size=32, sampler=valid_sampler)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

sample = next(iter(train_loader))


######################################################################
# Visualize a few images
# ^^^^^^^^^^^^^^^^^^^^^^
# Let's visualize a few training images so as to understand the data
# augmentations.

# Get a batch of training data
sample = next(iter(train_loader))
inputs, classes = sample['image'], sample['labels']

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)
imshow(out)

dataloaders = {'train': train_loader, 'val': validation_loader}
dataset_sizes = {'train': 11700, 'val': 1300}

model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 30)
model_ft = model_ft.to(device)

classes = animals_dataset.labels_to_idx
criterion = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights(classes)).to(device))

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

pred_list, gt_list = [], []

# model_ft.load_state_dict(torch.load("./models/Resnet50_finetuning/model_resnet_affine_ep20.net"))

epochs = 10
since = time.time()
for epoch in range(1, epochs + 1):
    train(model_ft, device, train_loader, optimizer_ft, epoch, criterion, classes)
    pred_list, gt_list = test(model_ft, device, validation_loader)
    print("Time Taken for epoch%d: %d sec"%(epoch, time.time()-since))
    since = time.time()
    eval_results_epoch(pred_list, gt_list, classes)
    if epoch % 5 == 0:
        torch.save(model_ft.state_dict(), "./models/Resnet50_finetuning/model_resnet_affine_weighted_class_ep%d.net" % epoch)

# torch.save(model_ft.state_dict(), "./models/model_inception_v3_pil_ep15.net")

#######################
# Testing an image

model_ft.eval()
data_iter = iter(test_dataset)

view_predictions(data_iter, device, model_ft, animals_dataset)

test_out = evaluate_testset(model_ft, test_loader, device)

frame = pd.DataFrame(test_out, columns=animals_dataset.labels_to_idx)
frame.head()
frame.shape

sample_submission = pd.read_csv('./meta-data_new/meta-data/sample_submission.csv')
sample_submission.head()
sample_submission.columns

frame['image_id'] = sample_submission['image_id']
frame = frame.loc[:, sample_submission.columns]
frame.head()
frame.to_csv('submissions/submission_resnet50_epoch10_affine_classweights.csv', index=False)