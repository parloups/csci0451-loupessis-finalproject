"""
Parker Loupessis Final Project
2nd Model, add in player direction and speed
"""

import torch
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.nn as nn
from torch.nn import ReLU
from torchinfo import summary
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# load tensors
data = torch.load("data/tensors.pt")
X_train = data["X_train"]
X_val = data["X_val"]
X_test = data["X_test"]
y_train = data["y_train"]
y_val = data["y_val"]
y_test = data["y_test"]

# convert to datatypes that will work for the model
X_train = X_train.to(torch.float32)
X_val = X_val.to(torch.float32)
X_test = X_test.to(torch.float32)
y_train = y_train.to(torch.long)
y_val = y_val.to(torch.long)
y_test = y_test.to(torch.long)

# create data loaders
train_set = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
val_set = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_set, batch_size=8, shuffle=False)
test_set = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_set, batch_size=8, shuffle=False)

class RouteClassifier(nn.Module):
  def __init__(self, input_size=6, snap_size=2, hidden_size=64, num_layers=2, num_classes=4, dropout=0.3):
    super().__init__()

    self.snap_encoder = nn.Sequential(
        nn.Linear(snap_size, 16),
        nn.ReLU()
    )

    self.gru = nn.GRU(input_size=input_size, 
                      hidden_size=hidden_size,
                      num_layers=num_layers,
                      batch_first=True,
                      dropout=dropout if num_layers > 1 else 0
    )
    
    self.pipeline = torch.nn.Sequential(
        nn.Linear(hidden_size + 16, 128),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(64, num_classes)
    )
  
  def forward(self, x):
    lengths = x[:, 0, -1].long()
    snap_location = x[:,0,2:4]
    x = x[:,:, [0,1,6,7,8,9]]

    snap_encoded = self.snap_encoder(snap_location)

    packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
    out, hidden = self.gru(packed)
    out, _ = pad_packed_sequence(out, batch_first=True)
    last_hidden = hidden[-1]

    combined = torch.cat([last_hidden, snap_encoded], dim=1)
    return self.pipeline(combined)
  
# check model complexity
X_batch = next(iter(train_loader))[0]
model = RouteClassifier()
summary(model, input_data=X_batch)

# function to evaluate the model
def evaluate(model, data_loader): 
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

# plot the confusion matrix 
LABELS = ["Go", "In", "Out", "Curl"]
def plot_confusion_mat(conf_mat, ax, title = "Confusion Matrix"):
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

# function to train the model 
def train(model, k_epochs=1, lr = 0.001):
  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  train_acc = []
  val_acc = []
  train_loss = []
  val_loss = []

  for epoch in range(k_epochs):
    for i, data in enumerate(train_loader):
      X, y = data
      optimizer.zero_grad()
      y_pred = model(X)
      loss = loss_fn(y_pred, y)
      loss.backward()
      optimizer.step()

    train_a, train_l, train_cm = evaluate(model, data_loader=train_loader)
    val_a, val_l, val_cm = evaluate(model, data_loader=val_loader)

    train_acc.append(train_a)
    val_acc.append(val_a)
    train_loss.append(train_l)
    val_loss.append(val_l)

  return train_acc, train_loss, train_cm, val_acc, val_loss, val_cm

# train
train_acc, train_loss, train_cm, val_acc, val_loss, val_cm = train(model, k_epochs = 200, lr = 0.001)

# plot loss and accuracy over training
fig, axarr = plt.subplots(1, 2, figsize = (8, 3.5))

ax = axarr[0]
ax.plot(train_loss, color = "black", label = "Training")
ax.plot(val_loss, color = "firebrick", label = "Validation")
ax.set_title("Cross-Entropy Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")

ax = axarr[1]
ax.plot(train_acc, color = "black", label = "Training")
ax.plot(val_acc, color = "firebrick", label = "Validation")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.set_title("Classification Accuracy")
ax.set(ylim = (0, 1))
plt.tight_layout()
l = ax.legend()

# plot confusion matrix after training
fig, axarr = plt.subplots(1, 2, figsize = (14, 7))
ax = axarr[0]
plot_confusion_mat(train_cm, ax, title = f"Training Confusion Matrix (acc = {train_acc[-1]:.2%})")
ax = axarr[1]
plot_confusion_mat(val_cm, ax, title = f"Validation Confusion Matrix (acc = {val_acc[-1]:.2%})")
plt.show()