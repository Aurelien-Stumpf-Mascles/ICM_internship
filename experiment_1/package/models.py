import torch
import torch.nn as nn

# Construct a model with one layer
class Model_MLP(nn.Module):
    
    def __init__(self, layer_size_list, non_linearity, initial_weights, type = 'regression'):
        super().__init__()
        input_size = layer_size_list[0]
        output_size = layer_size_list[-1]
        hidden_size_list = layer_size_list[1:-1]
        self.num_layers = len(hidden_size_list) + 1
        self.type = type
        if non_linearity == 'relu':
            self.non_linearity = torch.relu
        elif non_linearity == 'sigmoid':
            self.non_linearity = torch.sigmoid
        elif non_linearity == 'tanh':
            self.non_linearity = torch.tanh
        elif callable(non_linearity):
            self.non_linearity = non_linearity
        else:
            self.non_linearity = None
        self.initial_weights = initial_weights
        
        if self.num_layers == 1:
            self.l1 = nn.Linear(input_size, output_size)
        else:
            self.l1 = nn.Linear(input_size, hidden_size_list[0])
            for i in range(self.num_layers-2):
                setattr(self, 'l{}'.format(i+2), nn.Linear(hidden_size_list[i], hidden_size_list[i+1]))
            setattr(self, 'l{}'.format(self.num_layers), nn.Linear(hidden_size_list[-1], output_size))

        #initialize weights and biases
        for i in range(self.num_layers):
            if initial_weights == 'normal':
                getattr(self, 'l{}'.format(i+1)).weight.data.normal_(0.0, 1)
            elif initial_weights == 'xavier':
                nn.init.xavier_normal_(getattr(self, 'l{}'.format(i+1)).weight)
            elif initial_weights == 'kaiming':
                nn.init.kaiming_normal_(getattr(self, 'l{}'.format(i+1)).weight,nonlinearity=non_linearity)
            elif initial_weights == 'zero':
                getattr(self, 'l{}'.format(i+1)).weight.data.fill_(0.0)
            elif initial_weights == 'one':
                getattr(self, 'l{}'.format(i+1)).weight.data.fill_(1.0)
            elif initial_weights == 'uniform':
                getattr(self, 'l{}'.format(i+1)).weight.data.uniform_(-1.0, 1.0)
            elif initial_weights == 'xavier_uniform':
                nn.init.xavier_uniform_(getattr(self, 'l{}'.format(i+1)).weight)
            elif initial_weights == 'kaiming_uniform':
                nn.init.kaiming_uniform_(getattr(self, 'l{}'.format(i+1)).weight,nonlinearity=non_linearity)
            getattr(self, 'l{}'.format(i+1)).bias.data.fill_(0.0)
        
    def forward(self, inputs):
        if self.num_layers == 1:
            outputs = getattr(self, 'l1')(inputs)
        else:
            outputs = getattr(self, 'l1')(inputs)
            if self.non_linearity is not None:
                outputs = self.non_linearity(outputs)
            for i in range(self.num_layers-2):
                outputs = getattr(self, 'l{}'.format(i+2))(outputs)
                if self.non_linearity is not None:
                    outputs = self.non_linearity(outputs)
            outputs = getattr(self, 'l{}'.format(self.num_layers))(outputs)

        if self.type == "classification":
            outputs = nn.LogSoftmax(dim=1)(outputs)

        return outputs

# Construct a model with one layer
class Quadratic_MLP(nn.Module):
    
    def __init__(self, layer_size_list, non_linearity, initial_weights, type = 'regression'):
        super().__init__()
        self.input_size = layer_size_list[0]
        self.output_size = layer_size_list[-1]
        self.hidden_size_list = layer_size_list[1:-1]
        self.num_layers = len(self.hidden_size_list) + 1
        self.type = type
        if non_linearity == 'relu':
            self.non_linearity = torch.relu
        elif non_linearity == 'sigmoid':
            self.non_linearity = torch.sigmoid
        elif non_linearity == 'tanh':
            self.non_linearity = torch.tanh
        elif callable(non_linearity):
            self.non_linearity = non_linearity
        else:
            self.non_linearity = None
        self.initial_weights = initial_weights
        
        if self.num_layers == 1:
            self.l1 = nn.Parameter(torch.zeros(self.input_size, self.output_size))
        else:
            self.l1 = nn.Parameter(torch.zeros(self.input_size, self.hidden_size_list[0] * self.input_size))
            for i in range(self.num_layers-2):
                setattr(self, 'l{}'.format(i+2), nn.Linear(self.hidden_size_list[i], self.hidden_size_list[i+1]))
            setattr(self, 'l{}'.format(self.num_layers), nn.Linear(self.hidden_size_list[-1], self.output_size))

        #initialize weights and biases
        if initial_weights == 'normal':
            getattr(self, 'l{}'.format(1)).data.normal_(0.0, 1)
        elif initial_weights == 'xavier':
            nn.init.xavier_normal_(getattr(self, 'l{}'.format(1)))
        elif initial_weights == 'kaiming':
            nn.init.kaiming_normal_(getattr(self, 'l{}'.format(1)),nonlinearity=non_linearity)
        elif initial_weights == 'zero':
            getattr(self, 'l{}'.format(1)).data.fill_(0.0)
        elif initial_weights == 'one':
            getattr(self, 'l{}'.format(1)).data.fill_(1.0)
        elif initial_weights == 'uniform':
            getattr(self, 'l{}'.format(1)).data.uniform_(-1.0, 1.0)
        elif initial_weights == 'xavier_uniform':
            nn.init.xavier_uniform_(getattr(self, 'l{}'.format(1)))
        elif initial_weights == 'kaiming_uniform':
            nn.init.kaiming_uniform_(getattr(self, 'l{}'.format(1)),nonlinearity=non_linearity)

        for i in range(1,self.num_layers):
            if initial_weights == 'normal':
                getattr(self, 'l{}'.format(i+1)).weight.data.normal_(0.0, 1)
            elif initial_weights == 'xavier':
                nn.init.xavier_normal_(getattr(self, 'l{}'.format(i+1)).weight)
            elif initial_weights == 'kaiming':
                nn.init.kaiming_normal_(getattr(self, 'l{}'.format(i+1)).weight,nonlinearity=non_linearity)
            elif initial_weights == 'zero':
                getattr(self, 'l{}'.format(i+1)).weight.data.fill_(0.0)
            elif initial_weights == 'one':
                getattr(self, 'l{}'.format(i+1)).weight.data.fill_(1.0)
            elif initial_weights == 'uniform':
                getattr(self, 'l{}'.format(i+1)).weight.data.uniform_(-1.0, 1.0)
            elif initial_weights == 'xavier_uniform':
                nn.init.xavier_uniform_(getattr(self, 'l{}'.format(i+1)).weight)
            elif initial_weights == 'kaiming_uniform':
                nn.init.kaiming_uniform_(getattr(self, 'l{}'.format(i+1)).weight,nonlinearity=non_linearity)
            getattr(self, 'l{}'.format(i+1)).bias.data.fill_(0.0)
        
    def forward(self, inputs):
        batch_size = inputs.shape[0]
        if self.num_layers == 1:
            input_dim = inputs.shape[1]
            inputsW = torch.mm(inputs, self.l1).reshape((batch_size, -1, input_dim))
            outputs = torch.matmul(inputsW,inputs.unsqueeze(-1)).squeeze(-1)
        else:
            input_dim = inputs.shape[1]
            inputsW = torch.mm(inputs, self.l1).reshape((batch_size, -1, input_dim))
            outputs = torch.matmul(inputsW,inputs.unsqueeze(-1)).squeeze(-1)
            if self.non_linearity is not None:
                outputs = self.non_linearity(outputs)
            for i in range(self.num_layers-2):
                outputs = getattr(self, 'l{}'.format(i+2))(outputs)
                if self.non_linearity is not None:
                    outputs = self.non_linearity(outputs)
            outputs = getattr(self, 'l{}'.format(self.num_layers))(outputs)

        if self.type == "classification":
            outputs = nn.LogSoftmax(dim=1)(outputs)

        return outputs