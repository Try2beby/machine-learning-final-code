import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from sklearn.model_selection import train_test_split
import numpy as np
import os
import pandas as pd

dataDir = "../../data/ml-1m/"

# %matplotlib inline
from matplotlib_inline import backend_inline
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(
        self,
        num_classes=5,
        epoch=10,
        lr=0.001,
        tol=1e-4,
        verbose=False,
        print_every=100,
    ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(44, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.userId_embedding = nn.Embedding(6040 + 1, 10)
        self.movieId_embedding = nn.Embedding(3952 + 1, 10)
        self.epoch = epoch
        self.verbose = verbose
        self.print_every = print_every
        self.lr = lr
        self.tol = tol
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.loss_record = []
        self.accuracy_record = []
        self.load_data()

    def forward(self, x):
        # print(x.shape)
        user_embeds = self.userId_embedding(x[:, :, 0].long())
        movie_embeds = self.movieId_embedding(x[:, :, 1].long())
        # print(user_embeds.shape, movie_embeds.shape, x[:, :, 2:].shape)
        x = torch.cat([user_embeds, movie_embeds, x[:, :, 2:]], dim=2)
        # print(x.shape)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

    def load_data(self):
        # read data from csv
        data = pd.read_csv(os.path.join(dataDir, "data.csv"))

        # drop_cols = ["UserID", "MovieID", "Title", "Zip-code"]
        drop_cols = ["Title", "Zip-code"]

        # Split data into features and target variable
        X = data.drop("Rating", axis=1)
        X = X.drop(drop_cols, axis=1)
        y = data["Rating"]

        # turns y to one-hot encoding
        y = pd.get_dummies(y)

        # split data into train, validation and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

        batch_size = 64
        train_data = TensorDataset(
            torch.tensor(X_train.values, dtype=torch.float),
            torch.tensor(y_train.values, dtype=torch.float),
        )
        self.train_loader = DataLoader(train_data, batch_size=batch_size)
        test_data = TensorDataset(
            torch.tensor(X_test.values, dtype=torch.float),
            torch.tensor(y_test.values, dtype=torch.float),
        )
        # split test data into validation and test
        test_data, val_data = torch.utils.data.random_split(
            test_data,
            [int(0.5 * len(test_data)), len(test_data) - int(0.5 * len(test_data))],
            generator=torch.Generator().manual_seed(42),
        )
        self.test_loader = DataLoader(test_data, batch_size=batch_size)
        self.val_loader = DataLoader(val_data, batch_size=batch_size)

    def train(self, pretrained=False):
        if pretrained == False:
            self.init_weights()
        else:
            self.read_model()
        train_loader = self.train_loader

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        epoch_loss_prev = -1
        for epoch in range(self.epoch):
            epoch_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs = inputs.unsqueeze(1)
                inputs = inputs.to(self.device)
                labels = labels.float().to(self.device)
                optimizer.zero_grad()
                outputs = self(inputs).squeeze(1)
                # check the shape of outputs and labels
                # print(outputs.shape, labels.shape)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                # if self.verbose and i % self.print_every == 0:
                #     print(
                #         "Epoch: {}, batch: {}, loss: {}".format(epoch, i, loss.item())
                #     )
            epoch_loss /= len(train_loader)
            self.loss_record.append(epoch_loss)
            self.evaluate_on_train_and_val()

            # if self.verbose and epoch % self.print_every == 0:
            if self.verbose:
                print("Epoch: {}, loss: {}".format(epoch, epoch_loss))

            # if abs(epoch_loss - epoch_loss_prev) < self.tol:
            #     print("Converged at epoch: {}".format(epoch))
            #     break

            if epoch == self.epoch - 1:
                print("max epoch reached")

            epoch_loss_prev = epoch_loss

        print("Training completed")
        torch.save(self.state_dict(), "model.pth")
        # save the loss record and accuracy record
        np.save("loss_record.npy", np.array(self.loss_record))
        np.save("accuracy_record.npy", np.array(self.accuracy_record))

    def evaluate(self, loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in loader:
                inputs, labels = data
                inputs = inputs.unsqueeze(1)
                inputs = inputs.to(self.device)
                _, labels = torch.max(labels.long().to(self.device), dim=1)

                outputs = self(inputs).squeeze(1)
                # print(outputs.data[:5], outputs.shape)
                # predict a class
                _, predicted = torch.max(outputs.data, dim=1)
                # print(predicted.data[:5])
                # print(predicted.shape, labels.shape)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # print(labels.size(0), (predicted == labels).sum().item())
                # break
        # print(total, correct)
        return total, correct

    def evaluate_on_train_and_val(self):
        train_total, train_correct = self.evaluate(self.train_loader)
        val_total, val_correct = self.evaluate(self.val_loader)
        self.accuracy_record.append(
            [train_correct / train_total, val_correct / val_total]
        )
        print(
            "Accuracy of the network on the %d train data: %.2f %%"
            % (train_total, 100 * train_correct / train_total)
        )
        print(
            "Accuracy of the network on the %d val data: %.2f %%"
            % (val_total, 100 * val_correct / val_total)
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def read_model(self):
        self.load_state_dict(torch.load("model.pth"))

    def plot(self):
        backend_inline.set_matplotlib_formats("svg")

        loss_record = np.load("loss_record.npy")
        accuracy_record = np.load("accuracy_record.npy")
        plt.plot(loss_record)
        # set xticks
        plt.xticks(np.arange(0, self.epochs, 5))
        plt.xlim(0, self.epochs)
        plt.title("Loss")
        plt.show()
        plt.plot(accuracy_record[:, 0], label="train")
        plt.plot(accuracy_record[:, 1], label="val")
        # set xticks
        plt.xticks(np.arange(0, self.epochs, 5))
        plt.xlim(0, self.epochs)
        plt.title("Accuracy")
        plt.legend()
        plt.show()
