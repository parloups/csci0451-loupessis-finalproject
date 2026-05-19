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
  def __init__(self, model, second_labels=False):
    """
        Args:
            model:         instantiated RouteClassifier model to train and evaluate
            second_labels: if True, use 15 route classes instead of 4
      """
    self.model = model
    self.second_labels = second_labels

  # code from class notes
  def evaluate(self, model, data_loader):
    """
        Evaluates model performance on a given dataset.

        Args:
            model:       model to evaluate
            data_loader: DataLoader for the dataset to evaluate on

        Returns:
            acc:      overall classification accuracy
            loss:     total cross entropy loss across all batches
            conf_mat: confusion matrix of shape (num_classes, num_classes)
                      where conf_mat[i, j] is the number of samples with
                      true label i predicted as label j
      """
    if self.second_labels:
      # if second labels initalize as 15x15
      conf_mat = torch.zeros((15, 15), dtype = torch.int32)
    else:
      # if first labels initalize as 4x4
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

  # code from class notes
  def plot_confusion_mat(self, conf_mat, ax, title = "Confusion Matrix"):
    """
        Plots a confusion matrix as a heatmap with class labels and count annotations.

        Args:
            conf_mat: confusion matrix tensor of shape (num_classes, num_classes)
            ax:       matplotlib axes to plot on
            title:    plot title
      """
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

  # code similar from class notes
  def train(self, X, y, augmentation_count=0, k_epochs=1, lr=0.001, random_state=1):
    """
        Trains the model on all provided data for a fixed number of epochs.
        Used for final model training after hyperparameters are selected via CV.
        Saves the weights that achieved the lowest training loss as a safeguard
        against loss spikes near the end of training.

        Args:
            X:                 input tensor of shape (num_samples, max_frames, 11)
            y:                 label tensor of shape (num_samples,)
            augmentation_count: target count per class for augmentation (0 = no augmentation)
            k_epochs:          number of epochs to train for
            lr:                learning rate for Adam optimizer
            random_state:      seed for reproducibility

        Returns:
            train_acc:  list of training accuracy per epoch
            train_loss: list of training loss per epoch
            train_cm:   confusion matrix from final evaluation
      """
    torch.manual_seed(random_state)

    # augment training data to address class imbalance if target count provided
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

      # save weights whenever training loss improves
      if train_l < best_l:
        best_l = train_l
        best_weights = copy.deepcopy(self.model.state_dict())

    # restore best weights if final epoch was not the best
    if train_l > best_l:
      self.model.load_state_dict(best_weights)
      _, _, train_cm = self.evaluate(self.model, data_loader=train_loader)
    
    return train_acc, train_loss, train_cm

  def cv_train(self, model_type, X, y, augmentation_count=0, k_epochs=1, lr=0.001, patience=100, folds=5, random_state=1):
    """
        Trains and evaluates the model using stratified k-fold cross validation.
        Used for hyperparameter selection and honest performance estimation.

        Each fold:
            - Splits data into train/val maintaining class proportions (stratified)
            - Augments training data only — validation data is never augmented
            - Trains with early stopping based on validation accuracy
            - Plots loss and accuracy curves and confusion matrices

        Args:
            model_type:        class (not instance) of the model to instantiate each fold
            X:                 input tensor of shape (num_samples, max_frames, 11)
            y:                 label tensor of shape (num_samples,)
            augmentation_count: target count per class for augmentation (0 = no augmentation)
            k_epochs:          maximum number of epochs per fold
            lr:                learning rate for Adam optimizer
            patience:          number of epochs without val accuracy improvement before stopping
            folds:             number of CV folds
            random_state:      seed for reproducibility of splits and weight initialization

        Returns:
            cv_train_acc:  list of final training accuracy per fold
            cv_train_loss: list of final training loss per fold
            cv_train_cm:   list of training confusion matrices per fold
            cv_val_acc:    list of final validation accuracy per fold
            cv_val_loss:   list of final validation loss per fold
            cv_val_cm:     list of validation confusion matrices per fold
      """
    torch.manual_seed(random_state)
    cv_train_acc, cv_train_loss, cv_train_cm = [], [], []
    cv_val_acc, cv_val_loss, cv_val_cm = [], [], []

    # stratified splits ensure each fold has proportional class representation
    cv_folds = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    for fold, (train_idx, val_idx) in enumerate(cv_folds.split(X, y)):
      X_train, y_train = X[train_idx], y[train_idx]
      X_val, y_val = X[val_idx], y[val_idx]

      # augment training data only
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

      # reinitialize model each fold to prevent weight leakage between folds
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

        # save best weights when val accuracy improves
        # code suggested by AI
        if val_a > best_val_a:
          best_val_a = val_a
          epochs_no_imporve = 0
          best_weights = copy.deepcopy(current_model.state_dict())
        else:
          epochs_no_imporve += 1

        if epochs_no_imporve > patience:
          # restore best weights before final evaluation
          # code suggested by AI
          current_model.load_state_dict(best_weights)
          train_a, train_l, train_cm = self.evaluate(current_model, data_loader=train_loader)
          val_a, val_l, val_cm = self.evaluate(current_model, data_loader=val_loader)
          print(f"Stopping fold {fold+1} at epoch {epoch}. Final val accuracy of {val_a:.2%}")
          break
      
      # plot loss and accuracy curves for this fold
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

      # plot confusion matrices for this fold
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