from tqdm import tqdm

from model.base import *


class MLP(BASE):
    def __init__(self, **kwargs):
        super().__init__(kwargs['lr'])
        layers = kwargs['layers']
        input_size = kwargs["n_features"] * kwargs["seq_len"]
        
        modules = [
            nn.Linear(input_size, layers[0]),
            nn.BatchNorm1d(layers[0]),
            nn.ReLU()
        ]
        prev = layers[0]
        
        for layer in layers[1:-1]:
            modules.extend([
                nn.Linear(prev, layer),
                nn.BatchNorm1d(layer),
                nn.ReLU()
                ])
            prev = layer
        modules.append(nn.Linear(prev, layers[-1]))
        
        self.mlp = nn.Sequential(*modules)
            
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.mlp(x)
        return x.reshape(x.shape[0], 1, 1)


class LSTM(BASE):
    def __init__(self, **kwargs):
        super().__init__(kwargs['lr'])
        input_size = kwargs["n_features"]
        hidden_size = kwargs['hidden_size']
        num_layers = kwargs['num_layers']
        seq_len = kwargs["seq_len"]
        
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                                  num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(seq_len * hidden_size, 1)
    
    def forward(self, x):
        x = self.lstm(x)[0]
        x = self.fc(x.reshape(x.shape[0], -1))
        return x.reshape(x.shape[0], 1, 1)