import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class Model(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(Model, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = RNN(embed_size,hidden_size,num_layers,1)
        self.sigmoid = nn.Sigmoid()
        self.mseloss = nn.MSELoss()

    def forward(self, x):

        x = self.embed(x)  # (N, W, D)

        x = self.rnn(x)  # (N, 1, 1)

        x = self.sigmoid(x)

        return x.squeeze()


if __name__ == "__main__":
    model = Model(vocab_size = 35000, embed_size = 300, hidden_size = 300,\
                               num_layers = 2, prev_days = 3)
    from data_loader import get_loader

    loader = get_loader("data", "stock.csv")
    for i, data in enumerate(loader):
        x = model(data[0])
        print(x)
        print(data[1])
        if i == 1:
            break