import torch


def save_model(path, model, epoch, trainingLosses, trainingAccuracies, validationLosses, validationAccuracies):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'trainingLosses': trainingLosses,
        'trainingAccuracies': trainingAccuracies,
        'validationLosses': validationLosses,
        'validationAccuracies': validationAccuracies
    }, path + "/checkpoint.pth")

def load_model(path):
    return torch.load(path + "/checkpoint.pth")