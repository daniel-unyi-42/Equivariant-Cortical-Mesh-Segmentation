import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.functional import softmax, one_hot
from sklearn.metrics import confusion_matrix

def get_dice_loss(y_pred, y_true):
  num_classes = y_pred.shape[1]
  y_true = one_hot(y_true, num_classes=num_classes)
  y_pred = softmax(y_pred, dim=1)
  loss = 0.0
  for i in range(num_classes):
    loss += -2.0 * torch.sum(y_true[:,i] * y_pred[:,i]) / torch.sum(y_true[:,i] + y_pred[:,i])
  loss /= num_classes
  return loss

def get_Jaccard_index(y_pred, y_true):
  y_pred = y_pred.cpu().detach().numpy()
  y_true = y_true.cpu().detach().numpy()
  cm = confusion_matrix(y_true, y_pred.argmax(axis=1))
  TP = np.diag(cm)
  FP = np.sum(cm, axis=0) - TP
  FN = np.sum(cm, axis=1) - TP
  return TP / (TP + FP + FN)

def plot_learning_curve(val_losses, val_IoUs):
  plt.plot(val_losses)
  plt.xlabel('Epochs')
  plt.ylabel('Validation loss')
  plt.show()
  plt.plot(val_IoUs)
  plt.xlabel('Epochs')
  plt.ylabel('Validation IoU')
  plt.show()
