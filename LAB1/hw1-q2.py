#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from matplotlib import pyplot as plt

import utils


# Q2.1
class LogisticRegression(nn.Module):

    def __init__(self, n_classes, n_features, **kwargs):

        super(LogisticRegression, self).__init__()
        self.layer = nn.Linear(n_features, n_classes)
        #self.activation = nn.Sigmoid()

    def forward(self, x, **kwargs):
        x = self.layer(x)
        x = self.activation(x)
        return x


# Q2.2
class FeedforwardNetwork(nn.Module):
    def __init__(
            self, n_classes, n_features, hidden_size, layers,
            activation_type, dropout, **kwargs):

        super(FeedforwardNetwork, self).__init__()

        #self.layers = nn.Sequential()

        self.f1 = nn.Linear(n_features, hidden_size)

        self.dropout =  nn.Dropout(p=dropout)

        if activation_type == 'relu':
            self.activation = nn.ReLU()

        elif activation_type == 'tanh':
            self.activation = nn.Tanh()

        self.f2 = nn.Linear(hidden_size, n_classes)
        

    def forward(self, x, **kwargs):

        out = self.f1(x)
        out = self.activation(out)
        out = self.f2(out)
        return out


def train_batch(X, y, model, optimizer, criterion, **kwargs):

    optimizer.zero_grad()
    y_hat = model(X)
    loss = criterion(y_hat, y)
    loss.backward()
    optimizer.step()
    return loss.item()  
    


def predict(model, X):
    """X (n_examples x n_features)"""
    scores = model(X)  # (n_examples x n_classes)
    predicted_labels = scores.argmax(dim=-1)  # (n_examples)
    return predicted_labels


def evaluate(model, X, y):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    """
    model.eval()
    y_hat = predict(model, X)
    n_correct = (y == y_hat).sum().item()
    n_possible = float(y.shape[0])
    model.train()
    return n_correct / n_possible


def plot(epochs, plottable, ylabel='', name=''):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.savefig('%s.pdf' % (name), bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-batch_size', default=1, type=int,
                        help="Size of training batch.")
    parser.add_argument('-learning_rate', type=float, default=0.01)
    parser.add_argument('-l2_decay', type=float, default=0)
    parser.add_argument('-hidden_sizes', type=int, default=100)
    parser.add_argument('-layers', type=int, default=1)
    parser.add_argument('-dropout', type=float, default=0.3)
    parser.add_argument('-activation',
                        choices=['tanh', 'relu'], default='relu')
    parser.add_argument('-optimizer',
                        choices=['sgd', 'adam'], default='sgd')
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    data = utils.load_classification_data()
    dataset = utils.ClassificationDataset(data)
    train_dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True)

    dev_X, dev_y = dataset.dev_X, dataset.dev_y
    test_X, test_y = dataset.test_X, dataset.test_y

    n_classes = torch.unique(dataset.y).shape[0]  # 10
    n_feats = dataset.X.shape[1]

    # initialize the model
    if opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = FeedforwardNetwork(
            n_classes,
            n_feats,
            opt.hidden_size,
            opt.layers,
            opt.activation,
            opt.dropout
        )

    # get an optimizer
    optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}

    optim_cls = optims[opt.optimizer]
    optimizer = optim_cls(
        model.parameters(),
        lr=opt.learning_rate,
        weight_decay=opt.l2_decay)

    # get a loss criterion
    criterion = nn.CrossEntropyLoss()

    # training loop
    epochs = torch.arange(1, opt.epochs + 1)
    train_mean_losses = []
    valid_accs = []
    train_losses = []
    for ii in epochs:
        print('Training epoch {}'.format(ii))
        for X_batch, y_batch in train_dataloader:
            loss = train_batch(
                X_batch, y_batch, model, optimizer, criterion)
            train_losses.append(loss)

        mean_loss = torch.tensor(train_losses).mean().item()
        print('Training loss: %.4f' % (mean_loss))

        train_mean_losses.append(mean_loss)
        valid_accs.append(evaluate(model, dev_X, dev_y))
        print('Valid acc: %.4f' % (valid_accs[-1]))

    print('Final Test acc: %.4f' % (evaluate(model, test_X, test_y)))
    # plot
    if opt.model == "logistic_regression":
        config = "{}-{}".format(opt.learning_rate, opt.optimizer)
    else:
        config = "{}-{}-{}-{}-{}-{}-{}".format(opt.learning_rate, opt.hidden_size, opt.layers, opt.dropout, opt.activation, opt.optimizer, opt.batch_size)

    plot(epochs, train_mean_losses, ylabel='Loss', name='{}-training-loss-{}'.format(opt.model, config))
    plot(epochs, valid_accs, ylabel='Accuracy', name='{}-validation-accuracy-{}'.format(opt.model, config))


if __name__ == '__main__':
    main()
