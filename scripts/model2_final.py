"""
Parker Loupessis Final Project
Train and save model1 for testing data
"""
import torch
from torch.utils.data import TensorDataset, DataLoader
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.TrainEval import TrainEval
from src.RouteClassifier2 import RouteClassifier2

# load tensors
data = torch.load("data/tensors.pt")
X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

# ensure conversion to datatypes that will work for the model
X_train = X_train.to(torch.float32)
X_test = X_test.to(torch.float32)
y_train = y_train.to(torch.long)
y_test = y_test.to(torch.long)

# instantiate model
model = RouteClassifier2()
TrainEval = TrainEval(model, second_labels=False)

# train model
train_acc, train_loss, train_cm = TrainEval.train(X_train, y_train,
                                                  augmentation_count=0,
                                                  k_epochs=125,
                                                  lr=0.001,
                                                  random_state=512)

# save the model
torch.save(model.state_dict(), "models/model2.pt")

# evaluate on test data
model.eval()
test_set = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_set, batch_size=8, shuffle=False)
test_acc, test_loss, test_cm = TrainEval.evaluate(model, test_loader)