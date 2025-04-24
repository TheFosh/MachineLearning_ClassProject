import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from numpy.ma.extras import average
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.nn.functional import normalize
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def readFoodInfo(filename):
    df = pd.read_csv(filename)
    df.dropna()
    df['brand'] = df['brand'].astype('category')
    categories = list(df['brand'].cat.categories)
    df['brand'] = df['brand'].cat.codes
    X = df.iloc[:, 2:].copy().to_numpy()
    y = df['brand'].copy().to_numpy()

    # for i in range(X.shape[1]):
    #     avg = np.average(X.iloc[:, i])
    #     st = np.std(X.iloc[:, i])
    #
    #     X.iloc[:, i] = (df.iloc[:, i] - avg) / st
    #
    return X, y, categories


class FoodData(Dataset):
    def __init__(self):
        X, y, _ = readFoodInfo("Sub_Sandwiches_OSAT new.csv")

        self.X = torch.tensor(X, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.long)

        X, y, _ = readFoodInfo("Sub_Sandwiches_OSAT new.csv")

        self.X_valid = torch.tensor(X, dtype=torch.float)
        self.y_valid = torch.tensor(y, dtype=torch.long)

        self.len = len(self.X)

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return self.len


class FoodNetwork(nn.Module):
    def __init__(self):
        super(FoodNetwork, self).__init__()
        self.in_to_h1 = nn.Linear(41, 20)

        self.h1_to_out = nn.Linear(20, 7)

    def forward(self, x):
        x = F.relu(self.in_to_h1(x))

        return self.h1_to_out(x)


def trainNN(epochs=10, batch_size=16, lr=0.001, trained_network=None, save_file="foodNN.pt"):
    fd = FoodData()

    dl = DataLoader(fd, batch_size=batch_size, shuffle=True, drop_last=True)

    foodNN = FoodNetwork()
    if trained_network is not None:
        foodNN.load_state_dict(torch.load(trained_network))
        foodNN.train()

    loss_fn = CrossEntropyLoss()

    optimizer = torch.optim.Adam(foodNN.parameters(), lr=lr)

    running_loss = 0.0
    test_results = np.array([])
    for epoch in range(epochs):
        for _, data in enumerate(tqdm(dl)):
            X, y = data

            optimizer.zero_grad()

            output = foodNN(X)

            loss = loss_fn(output, y)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        with torch.no_grad():
            foodNN.eval()
            print(f"Running loss for epoch {epoch + 1} of {epochs}: {running_loss:.4f}")
            predictions = torch.argmax(foodNN(fd.X), dim=1)
            correct = (predictions == fd.y).sum().item()
            print(f"Accuracy on train set: {correct / len(fd.X):.4f}")
            predictions = torch.argmax(foodNN(fd.X_valid), dim=1)
            correct = (predictions == fd.y_valid).sum().item()
            print(f"Accuracy on validation set: {correct / len(fd.X_valid):.4f}")
            test_results = np.append(test_results, correct / len(fd.X_valid))
            foodNN.train()
        running_loss = 0.0
    torch.save(foodNN.state_dict(), save_file)
    return test_results


def network_result(trained_network="foodNN.pt"):
    # Get Test Data
    X, y, categories = readFoodInfo("Sub_Sandwiches_OSAT new.csv")

    # Load the ANN
    foodNN = FoodNetwork()
    foodNN.load_state_dict(torch.load(trained_network))
    foodNN.eval()

    X = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)

    with torch.no_grad():
        predictions = torch.argmax(foodNN(X), dim=1)
        correct = (predictions == y).sum().item()
        print(f"Accuracy on test set: {correct / len(X):.4f}")
    cm = confusion_matrix(np.array(y), np.array(predictions))
    disp = ConfusionMatrixDisplay(cm, display_labels=categories)
    disp.plot()
    plt.show()

    importances = pd.DataFrame(foodNN.feature_importances_, index=pd.DataFrame(X).columns[:])
    importances.plot.bar()
    plt.show()


# trainNN(epochs=100, batch_size=8)
network_result()
