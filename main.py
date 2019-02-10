import matplotlib
import torch
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import transforms

from Utils import save_model
from model import CNN
from train import train
from validate import validate

dataset_transformation = transforms.ToTensor()
trainSet = datasets.MNIST(root='./data', train=True, download=True, transform=dataset_transformation)
testSet = datasets.MNIST(root='./data', train=False, download=True, transform=dataset_transformation)

train_loader = torch.utils.data.DataLoader(trainSet, batch_size=64, shuffle=True, num_workers=2)
valid_loader = torch.utils.data.DataLoader(testSet, batch_size=64, shuffle=True, num_workers=2)

checkpoint_path = "./saved_models"

trainingLosses = []
trainingAccuracies = []
validationLosses = []
validationAccuracies = []

for i in range(10):
    model = CNN()

    print("staring iteration: ", i)

    trainLoss, trainAcc = train(model, train_loader)
    validLoss, validAcc = validate(model, train_loader)

    trainingLosses.append(trainLoss);
    trainingAccuracies.append(trainAcc);

    validationLosses.append(validLoss)
    validationAccuracies.append(validAcc)

    print("saving checkpoint ")
    save_model(checkpoint_path, model, trainingLosses, trainingAccuracies, validationLosses, validationAccuracies)

plt.subplot(2,2,1)
plt.plot(trainingLosses, len(trainingLosses))
plt.ylabel('Train Loss')
plt.xlabel('Epochs')

plt.subplot(2,2,2)
plt.plot(trainingAccuracies, len(trainingAccuracies))
plt.ylabel('Train Accuracy')
plt.xlabel('Epochs')

plt.subplot(2,2,1)
plt.plot(validationLosses, len(validationLosses))
plt.ylabel('Validation Loss')
plt.xlabel('Epochs')

plt.subplot(2,2,2)
plt.plot(validationAccuracies, len(validationAccuracies))
plt.ylabel('Validation Accuracy')
plt.xlabel('Epochs')

fig = matplotlib.pyplot.gcf()
fig.set_size_inches(12, 12)