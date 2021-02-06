import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self, num_classes=1, input_size=1, hidden_size=48, num_layers = 2 ,output_size = 1 ):
        super(LSTM,self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size = input_size, hidden_size=hidden_size,num_layers = num_layers, batch_first = True)
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, 100),
            nn.Linear(100,50),
            nn.Linear(50, output_size)
        )
        self.linear_1 = nn.Linear(hidden_size, 96)
        self.linear_2 = nn.Linear(96, 48)
        self.linear_3 = nn.Linear(48, output_size)

        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.xavier_uniform_(self.linear_3.weight)
        # nn.init.xavier_uniform_(self.layers.weight)

        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        output, (hidden, cell) = self.lstm(x, (h_0, c_0))
        # output = self.relu(output)
        output = self.linear_1(output)
        output = self.linear_2(output)
        output = self.linear_3(output)[:,-1,]

        return output
