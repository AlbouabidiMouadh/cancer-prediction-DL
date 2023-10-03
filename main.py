import warnings

import matplotlib

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
from PIL import Image

import torch
from torch.nn import BCELoss
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CyclicLR
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from glob import glob
from skimage.io import imread
from os import listdir

import time
import copy
from tqdm import tqdm_notebook as tqdm

run_training = False
retrain = False
find_learning_rate = False

df = pd.read_csv("input/data_csv/data.csv")

print(df.describe().T)

print(df.isnull().sum())

# df = df.dropna()


print(df.dtypes)

sns.countplot(data=df, x="diagnosis")
# plt.show()

print(df['diagnosis'].value_counts())

y = df["diagnosis"].values
print(np.unique(y))


X = df.drop(labels=["diagnosis", "id"], axis=1)
print(X.describe().T)

scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

labelencoder = LabelEncoder()
Y = labelencoder.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
print("train part: ", x_train.shape)
print("test part: ", x_test.shape)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(30, 16)  # Define the first fully connected (dense) layer with 30 input features and 16 output features
        self.fc2 = nn.Linear(16, 1)   # Define the second fully connected layer with 16 input features and 1 output feature

    def forward(self, x):
        x = F.relu(self.fc1(x))       # Apply a ReLU activation function to the output of the first layer
        x = F.dropout(x, p=0.2, training=self.training)  # Apply dropout regularization to the output
        x = torch.sigmoid(self.fc2(x))  # Apply a sigmoid activation function to the output of the second layer
        return x


model = Net()


criterion = BCELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)


x_train_tensor = torch.tensor(x_train, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

x_test_tensor = torch.tensor(x_test, dtype=torch.float32)

y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

num_epochs = 100
for epoch in range(num_epochs):
    model.train()  # Set the model in training mode
    for inputs, labels in train_loader:
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute the loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update the model's weights

    # Set the model in evaluation mode (no gradient computation) using model.eval()
    model.eval()

    # Initialize variables to keep track of total samples, correct predictions, validation loss, and accuracy
    total = 0
    correct = 0
    val_loss = 0.0

    # Iterate through the test_loader, which contains batches of testing data and labels
    for inputs, labels in test_loader:
        outputs = model(inputs)  # Forward pass: obtain model predictions for the current batch
        predicted = (outputs > 0.5).float()  # Apply a threshold (0.5) for binary classification
        total += labels.size(0)  # Increment the total number of samples by the batch size
        correct += (predicted == labels).sum().item()  # Count the number of correct predictions
        val_loss += criterion(outputs, labels).item()  # Calculate the validation loss

    # Calculate the accuracy as the percentage of correct predictions over the total samples
    accuracy = 100 * correct / total

    # Calculate the average validation loss by dividing by the number of batches
    val_loss /= len(test_loader)

    # Append the training and validation loss, as well as training and validation accuracy, to their respective lists
    train_losses.append(loss.item())  # Append the current training loss to the list
    val_losses.append(val_loss)  # Append the current validation loss to the list
    train_accuracies.append(100 * (1 - loss.item()))  # Append the current training accuracy to the list
    val_accuracies.append(accuracy)  # Append the current validation accuracy to the list

    # Print the current epoch's training loss, validation loss, and validation accuracy
    print(
        f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%')



plt.figure(figsize=(10, 5))

plt.plot(range(1, num_epochs + 1), train_losses, "y", label='Training loss')

plt.plot(range(1, num_epochs + 1 ), val_losses, 'r', label= 'Validation loss')

plt.title('Training and Validation Loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()


plt.figure(figsize=(10, 5))

plt.plot(range(1, num_epochs + 1), train_accuracies, 'y', label='Training accuracy')

plt.plot(range(1, num_epochs + 1), val_accuracies, 'r', label='Validation accuracy')

plt.title('Training and validation Accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy (%)')

plt.legend()

plt.show()

model.eval()

y_pred_tensor = []

with torch.no_grad():
    for inputs, _ in test_loader:
        outputs = model(inputs)
        y_pred_tensor.extend(outputs.cpu().numpy())


y_pred = (np.array(y_pred_tensor) > 0.5)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot= True)

plt.show()


