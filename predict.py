from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from cnn_model import Net
from dataload import *
from data_transform import *
import matplotlib.pyplot as plt
import warnings
import numpy as np
import torch
warnings.filterwarnings(action='ignore')

data_transforms = transforms.Compose([
        Rotate2d(),
        RandomCrop((224, 224)),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
data_transforms_test = transforms.Compose([
        RandomCrop((224, 224)),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

data_transform1 = transforms.Compose([ToTensor()])
animals_dataset = AnimalsDataset(filename='meta-data_new/meta-data/train.csv', root_dir='./train_new/train/', train=True,
                                 transform=data_transforms)
test_dataset = AnimalsDataset(filename='meta-data_new/meta-data/test.csv', root_dir='./test_new/test', train=False,
                              transform=data_transforms_test)

test_loader = DataLoader(test_dataset, batch_size=60)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

net = Net().to(device)
print(net)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

net.load_state_dict(torch.load("./models/model_ep50.net"))

net.eval()

#######################
# Testing an image
data_iter = iter(test_dataset)

plt.figure(figsize=(10, 10))
for i in range(12):
    sample = next(data_iter)
    data, labels = sample['image'], sample['labels']

    img = data
    img = torch.unsqueeze(img, 0)
    img = img.float().to(device)

    out = net(img)
    prediction = torch.nn.functional.softmax(out)
    predicted_label = animals_dataset.labels_to_idx[torch.nn.functional.softmax(out).argmax().item() ]
    print("Predicted Label:", predicted_label)
    print("Probability:", torch.nn.functional.softmax(out).max().item())

    plt.subplot(4, 3, i+1)
    plt.imshow(torch.squeeze(img).cpu().numpy().transpose(1, 2, 0))
    plt.title(predicted_label)
    plt.text(240, 240, torch.nn.functional.softmax(out).max().item())
    plt.axis('off')
plt.show()

# print(prediction)
# print(animals_dataset.labels_to_idx)

soft_out = []
for i, sample in enumerate(test_loader):
    data, labels = sample['image'].float().to(device), sample['labels'].float().to(device)
    out = net(data)
    print(i, out.size())
    soft_out.append(torch.nn.functional.softmax(out).cpu().detach().numpy())

print(len(soft_out))
print(type(soft_out[0]))
test_out = np.concatenate(soft_out, 0)
test_out.shape

frame = pd.DataFrame(test_out, columns= animals_dataset.labels_to_idx)
frame.head()
frame.shape
sample_submission = pd.read_csv('./meta-data_new/meta-data/sample_submission.csv')
sample_submission.head()
sample_submission.columns

frame['image_id'] = sample_submission['image_id']
frame = frame.loc[:, sample_submission.columns]
frame.head()
frame.to_csv('submission_1.csv', index=False)

#####################

# Visualizing the activation maps
'''
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
'''