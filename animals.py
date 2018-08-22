from torchvision import transforms, utils
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from cnn_model import Net, train, test
from dataload import *
from data_transform import *
from data_utils import *
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.transforms.functional
import warnings
warnings.filterwarnings(action='ignore')

data_transform = transforms.Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
data_transform1 = transforms.Compose([ToTensor()])
animals_dataset = AnimalsDataset(filename='meta-data_new/meta-data/train.csv', root_dir='./train_new/train/', train=True,
                                 transform=data_transform1)
test_dataset = AnimalsDataset(filename='meta-data_new/meta-data/test.csv', root_dir='./test_new/test', train=False,
                              transform=data_transform1)
# data_iter = iter(animals_dataset)
# sample = next(data_iter)
# data, labels = sample['image'], sample['labels']
# print(data.max(), data.min())


validation_split = 0.1
shuffle_dataset = True
random_seed=42

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
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

#
# for i, sample in enumerate(train_loader):
#     if i == 0:
#         data, labels = sample['image'], sample['labels']
#         print(data.shape)
#         print(labels.shape)
#         for j in range(3):
#             plt.figure()
#             plt.imshow(data[j].numpy().transpose(1, 2, 0))
#             plt.title(animals_dataset.labels_to_idx[labels[j].argmax().item()])
#             plt.show()
#         break

net = Net().to(device)
print(net)
# optimizer = optim.Adam(net.parameters())
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
epochs = 50

for i, sample in enumerate(train_loader):
    if i == 0:
        data, labels = sample['image'].float().to(device), sample['labels'].float().to(device)
        out = net(data)
        print(out.size())
        break

net.load_state_dict(torch.load("./models/model_ep50.net"))

# for epoch in range(1, epochs + 1):
#     train(net, device, train_loader, optimizer, epoch)
#     test(net, device, validation_loader)


# torch.save(net.state_dict(), "./models/model_ep50.net")

# test(net, device, validation_loader)


data_iter = iter(animals_dataset)
sample = next(data_iter)
data, labels = sample['image'], sample['labels']

img = data
img = torch.unsqueeze(img, 0)
img = img.float().to(device)

out = net(img)

predicted_label = animals_dataset.labels_to_idx[torch.nn.functional.softmax(out).argmax().item() ]
gt_label = animals_dataset.labels_to_idx[labels.argmax().item()]
print("Predicted Label:", predicted_label)
print("GT Label:", gt_label)
print("Probability:", torch.nn.functional.softmax(out).max().item())

plt.figure()
plt.imshow(torch.squeeze(img).cpu().numpy().transpose(1, 2, 0))
plt.title(predicted_label)
plt.axis('off')
plt.show()

net.eval()

#######################
# Testing an image
data_iter = iter(animals_dataset)

plt.figure(figsize=(10, 10))
for i in range(12):
    sample = next(data_iter)
    data, labels = sample['image'], sample['labels']

    img = data
    img = torch.unsqueeze(img, 0)
    img = img.float().to(device)

    out = net(img)
    prediction = torch.nn.functional.softmax(out)
    gt_label = animals_dataset.labels_to_idx[labels.argmax().item()]
    predicted_label = animals_dataset.labels_to_idx[torch.nn.functional.softmax(out).argmax().item() ]
    print("Predicted Label:", predicted_label, " ", gt_label)
    print("Probability:", torch.nn.functional.softmax(out).max().item())

    plt.subplot(4, 3, i+1)
    plt.imshow(torch.squeeze(img).cpu().numpy().transpose(1, 2, 0))
    plt.title(predicted_label + " " + gt_label)
    plt.text(240, 240, torch.nn.functional.softmax(out).max().item())
    plt.axis('off')
plt.show()


print(prediction)
print(animals_dataset.labels_to_idx)
submit = pd.DataFrame(columns=animals_dataset.labels_to_idx)
submit

soft_out = []
for i, sample in enumerate(test_loader):
    data, labels = sample['image'].float().to(device), sample['labels'].float().to(device)
    out = net(data)
    print(out.size())
    soft_out.append(torch.nn.functional.softmax(out).cpu().detach().numpy())

print(len(soft_out))
torch.cat(soft_out, 0).size()

# .cpu().detach().numpy()
frame = pd.DataFrame(np.array(soft_out))
frame.head()

#####################
img1 = img

def show_image(img1):
    img1 = torch.squeeze(img1)
    img1 = img1.cpu().detach().numpy().transpose(1, 2, 0)
    print(img1.shape)
    plt.figure(figsize=(8, 8))
    for i in range(32):
        plt.subplot(4, 8, i + 1)
        plt.imshow(img1[:, :, i])
        # plt.title("filter{}".format(i + 1))
        plt.axis('off')
    plt.show()


def visualize_fmaps(net, img1, selected_layer):
    for index, layer in enumerate(net.modules()):
        print(index, layer)
        if index != 0:
            img1 = layer(img1)
        print(img1.size())
        if index == selected_layer:
            break
    show_image(img1)


visualize_fmaps(net, img1, 1)
visualize_fmaps(net, img1, 2)
visualize_fmaps(net, img1, 3)
