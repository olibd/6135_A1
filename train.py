import numpy as np

import torch
import torch.cuda as cuda
import torch.nn as nn
from torch.autograd import Variable


def train(model, train_loader):

    print("training started")

    device = torch.device("cpu")
    if(cuda.is_available()):
        device = torch.device("cuda")

    model.to(device)
    model.train()

    losses = []

    totalNumberItems = 0
    correctPredictions = 0

    for (images, targets) in train_loader:
        if (cuda.is_available()):
            images = images.to(device)
            targets = targets.to(device)

        images = Variable(images)
        target = Variable(targets)

        output = model(images)

        loss = nn.CrossEntropyLoss(output, target).to(device)
        losses.append(loss)
        loss.backward()

        predictedClasses = torch.argmax(output)
        totalNumberItems += targets.size(0)
        correctPredictions += predictedClasses.eq(targets.data).cpu().sum()

    print("training completed")

    return np.mean(losses), correctPredictions/totalNumberItems