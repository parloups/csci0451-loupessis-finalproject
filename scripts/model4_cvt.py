"""
Parker Loupessis Final Project
Train model4 using cross-validation
"""
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.TrainEval import TrainEval
from src.RouteClassifier4 import RouteClassifier4

# load tensors
data = torch.load("data/tensors2.pt")
X_train = data["X_train"]
y_train = data["y_train"]

# ensure conversion to datatypes that will work for the model
X_train = X_train.to(torch.float32)
y_train = y_train.to(torch.long)

# instantiate model
model = RouteClassifier4()
TrainEval = TrainEval(model, second_labels=True)

# run cross validation
train_acc, train_loss, train_cm, val_acc, val_loss, val_cm = TrainEval.cv_train(RouteClassifier4, X_train, y_train,
                                                                              augmentation_count=20,
                                                                              k_epochs=1000,
                                                                              lr=0.001,
                                                                              patience=200,
                                                                              folds=5,
                                                                              random_state=514
                                                                              )

# save results
torch.save({
    "train_acc":  train_acc,
    "train_loss": train_loss,
    "train_cm":   train_cm,
    "val_acc":    val_acc,
    "val_loss":   val_loss,
    "val_cm":     val_cm
}, "models/model4_cv_results.pt")