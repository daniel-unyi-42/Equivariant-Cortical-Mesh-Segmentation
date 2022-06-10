import torch
import torch.nn.functional as F
from metrics import get_dice_loss, get_Jaccard_index

def train_model(train_loader, model, optimizer):
  model.train()
  train_losses = []
  train_IoUs = []
  for train_data in train_loader:
    optimizer.zero_grad()
    y_pred = model(train_data)
    y_true = train_data.y
    train_loss = get_dice_loss(y_pred, y_true)    
    train_losses.append(train_loss)
    train_loss.backward()
    optimizer.step()
    train_IoU = get_Jaccard_index(y_pred, y_true)
    train_IoUs.append(train_IoU)
  return sum(train_losses) / len(train_losses), sum(train_IoUs) / len(train_IoUs)

def test_model(test_loader, model):
  model.eval()
  test_losses = []
  test_IoUs = []
  with torch.no_grad():
    for test_data in test_loader:
      y_pred = model(test_data)
      y_true = test_data.y
      test_loss = get_dice_loss(y_pred, y_true)
      test_losses.append(test_loss)
      test_IoU = get_Jaccard_index(y_pred, y_true)
      test_IoUs.append(test_IoU)
  return sum(test_losses) / len(test_losses), sum(test_IoUs) / len(test_IoUs)
