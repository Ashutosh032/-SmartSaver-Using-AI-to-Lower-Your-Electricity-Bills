import torch
import torch.nn as nn

class SmartSaverLSTM(nn.Module):
    def __init__(self, input_size=9, hidden_size=64, num_layers=2, output_steps=96, output_features=2):
        """
        LSTM for forecasting.
        input_size: Number of features
        output_steps: Number of time steps to predict (e.g., 24 hours * 4 = 96 steps)
        output_features: Number of features to predict (price, demand)
        """
        super(SmartSaverLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_steps = output_steps
        self.output_features = output_features
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        
        # Fully connected layer to map LSTM final state to the output sequence
        self.fc = nn.Linear(hidden_size, output_steps * output_features)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        
        # Reshape to (batch_size, output_steps, output_features)
        out = out.view(-1, self.output_steps, self.output_features)
        return out
