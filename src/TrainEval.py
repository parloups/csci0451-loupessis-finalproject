"""
Parker Loupessis Final Project
Class to train and evaluate the models
"""
import torch
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import copy
from sklearn.model_selection import StratifiedKFold
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.RouteAug import Route_Augmentation

class TrainEval:
  def __init__(self, model, second_labels):
    self.model = model
    self.second_labels = second_labels

  def evaluate(self, model, data_loader):
    if self.second_labels:
      conf_mat = torch.zeros((15, 15), dtype = torch.int32)
    else:
      conf_mat = torch.zeros((4, 4), dtype = torch.int32)

    loss_fn = nn.CrossEntropyLoss()
    loss = 0
    with torch.no_grad(): 
      for X_batch, y_batch in data_loader:
        scores = model(X_batch)
        loss += loss_fn(scores, y_batch).item()
        y_pred = torch.argmax(scores, dim = 1)
        for i in range(len(y_batch)):
          conf_mat[y_batch[i], y_pred[i]] += 1

    acc = torch.diag(conf_mat).sum() / conf_mat.sum()

    return acc, loss, conf_mat

  def plot_confusion_mat(self, conf_mat, ax, title = "Confusion Matrix"):
    if self.second_labels:
      LABELS = ['Post', 'Go', 'Wheel', 'Dig', 'Curl', 'Drag', 'Speed Out', 'Slot Fade', 'Slide', 'Slant', 'Crosser', 'Hitch', 
          'Whip', 'Out', 'Corner']
    else:
      LABELS = ["Go", "In", "Out", "Curl"]
    im = ax.imshow(conf_mat.cpu(), cmap = "Blues", origin = "upper")
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(torch.arange(len(LABELS)))
    ax.set_yticks(torch.arange(len(LABELS)))
    ax.set_xticklabels(list(LABELS))
    ax.set_yticklabels(list(LABELS))
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            text = ax.text(j, i, conf_mat[i, j].item(),
                           ha="center", va="center", color="black", size = 6)
    ax.set_title(title)
    ax.grid(False)

  def train(self, X, y, augmentation_count=0, k_epochs=1, lr=0.001, random_state=1):
    torch.manual_seed(random_state)
    augmenter = Route_Augmentation(augmentation_count)
    X_train_aug, y_train_aug = augmenter.augment(X, y)
    X_train_aug = X_train_aug.to(torch.float32)
    y_train_aug = y_train_aug.to(torch.long)
    train_set = TensorDataset(X_train_aug, y_train_aug)
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    best_l = float("inf")
    best_weights = None
    train_acc = []
    train_loss = []
    for epoch in range(k_epochs):
      for i, data in enumerate(train_loader):
        X, y = data
        optimizer.zero_grad()
        y_pred = self.model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

      train_a, train_l, train_cm = self.evaluate(self.model, data_loader=train_loader)
      train_acc.append(train_a)
      train_loss.append(train_l)

      if train_l < best_l:
        best_l = train_l
        best_weights = copy.deepcopy(self.model.state_dict())

    if train_l > best_l:
      self.model.load_state_dict(best_weights)
      _, _, train_cm = self.evaluate(self.model, data_loader=train_loader)
    
    return train_acc, train_loss, train_cm

  def cv_train(self, model_type, X, y, augmentation_count=0, k_epochs=1, lr=0.001, patience=100, folds=5, random_state=1):
    torch.manual_seed(random_state)
    cv_train_acc, cv_train_loss, cv_train_cm = [], [], []
    cv_val_acc, cv_val_loss, cv_val_cm = [], [], []

    cv_folds = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    for fold, (train_idx, val_idx) in enumerate(cv_folds.split(X, y)):
      X_train, y_train = X[train_idx], y[train_idx]
      X_val, y_val = X[val_idx], y[val_idx]

      augmenter = Route_Augmentation(augmentation_count)
      X_train_aug, y_train_aug = augmenter.augment(X_train, y_train)
      X_train_aug = X_train_aug.to(torch.float32)
      y_train_aug = y_train_aug.to(torch.long)

      train_set = TensorDataset(X_train_aug, y_train_aug)
      train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
      val_set = TensorDataset(X_val, y_val)
      val_loader = DataLoader(val_set, batch_size=8, shuffle=False)

      best_val_a = 0
      epochs_no_imporve = 0
      best_weights = None

      current_model = model_type()
      loss_fn = nn.CrossEntropyLoss()
      optimizer = torch.optim.Adam(current_model.parameters(), lr=lr, weight_decay=1e-3)

      train_acc = []
      val_acc = []
      train_loss = []
      val_loss = []

      for epoch in range(k_epochs):
        for _, data in enumerate(train_loader):
          X_batch, y_batch = data
          optimizer.zero_grad()
          y_pred = current_model(X_batch)
          loss = loss_fn(y_pred, y_batch)
          loss.backward()
          optimizer.step()

        train_a, train_l, train_cm = self.evaluate(current_model, data_loader=train_loader)
        val_a, val_l, val_cm = self.evaluate(current_model, data_loader=val_loader)

        train_acc.append(train_a)
        val_acc.append(val_a)
        train_loss.append(train_l)
        val_loss.append(val_l)

        if val_a > best_val_a:
          best_val_a = val_a
          epochs_no_imporve = 0
          best_weights = copy.deepcopy(current_model.state_dict())
        else:
          epochs_no_imporve += 1

        if epochs_no_imporve > patience:
          current_model.load_state_dict(best_weights)
          train_a, train_l, train_cm = self.evaluate(current_model, data_loader=train_loader)
          val_a, val_l, val_cm = self.evaluate(current_model, data_loader=val_loader)
          print(f"Stopping fold {fold+1} at epoch {epoch}. Final val accuracy of {val_a:.2%}")
          break
      
      # plot loss and accuracy over training
      fig, axarr = plt.subplots(1, 2, figsize = (8, 3.5))

      ax = axarr[0]
      ax.plot(train_loss, color = "black", label = "Training")
      ax.plot(val_loss, color = "firebrick", label = "Validation")
      ax.set_title(f"Fold {fold+1} Cross-Entropy Loss")
      ax.set_xlabel("Epoch")
      ax.set_ylabel("Loss")

      ax = axarr[1]
      ax.plot(train_acc, color = "black", label = "Training")
      ax.plot(val_acc, color = "firebrick", label = "Validation")
      ax.set_xlabel("Epoch")
      ax.set_ylabel("Accuracy")
      ax.set_title(f"Fold {fold+1} Classification Accuracy")
      ax.set(ylim = (0, 1))
      plt.tight_layout()
      l = ax.legend()

      # plot confusion matrix after training
      fig, axarr = plt.subplots(1, 2, figsize = (14, 7))
      ax = axarr[0]
      self.plot_confusion_mat(train_cm, ax, title = f"Fold {fold+1} Training Confusion Matrix (acc = {train_a:.2%})")
      ax = axarr[1]
      self.plot_confusion_mat(val_cm, ax, title = f"Fold {fold+1} Validation Confusion Matrix (acc = {val_a:.2%})")

      cv_train_acc.append(train_a)
      cv_train_loss.append(train_l)
      cv_train_cm.append(train_cm)
      cv_val_acc.append(val_a)
      cv_val_loss.append(val_l)
      cv_val_cm.append(val_cm)

    print(f"Average accuracy of {sum(cv_val_acc)/folds:.2%}")
    plt.show()
    return cv_train_acc, cv_train_loss, cv_train_cm, cv_val_acc, cv_val_loss, cv_val_cm