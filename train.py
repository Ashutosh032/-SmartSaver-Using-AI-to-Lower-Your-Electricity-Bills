import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys

# Add parent dir to path to import data module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.preprocessing import preprocess_data
from model.lstm import SmartSaverLSTM

def create_sequences(data, input_steps=192, output_steps=96):
    """
    Creates overlapping sequences for time-series forecasting.
    input_steps: 48 hours * 4 (15-min intervals) = 192
    output_steps: 24 hours * 4 = 96
    """
    xs, ys = [], []
    # Target features are 'demand_mw' (idx 0) and 'price_rs_per_kwh' (idx 1) in our preprocessing order
    # based on: ['demand_mw', 'price_rs_per_kwh', 'temperature_c', 'cloud_cover_pct', 'humidity_pct', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    # Wait, the scaler was only applied to the first 5. Let's make sure we have the correct indices.
    
    # Actually, let's just use column indices assuming the df is ordered.
    # We will pass the df values. In preprocessing, we merged and added columns.
    # The order: timestamp, demand_mw, price_rs_per_kwh, temperature_c, cloud_cover_pct, humidity_pct, hour_sin, hour_cos, day_sin, day_cos
    # So index 1 is demand, index 2 is price.
    
    for i in range(len(data) - input_steps - output_steps):
        x = data[i:(i + input_steps), 1:] # Skip timestamp
        y = data[(i + input_steps):(i + input_steps + output_steps), 1:3] # Only demand and price
        xs.append(x)
        ys.append(y)
        
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

def train_model():
    print("Pre-processing data for training...")
    # Fetch 1 month of dummy data for fast training demonstration
    df, scaler = preprocess_data('2023-01-01', '2023-01-31')
    
    data_values = df.values
    X, y = create_sequences(data_values)
    
    # Split into train and test
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SmartSaverLSTM(input_size=9, output_steps=96, output_features=2).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 5 # Reduced for synthetic demonstration
    print(f"Starting training on {device} for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.4f}")
        
    print("Training complete. Saving weights...")
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), 'smartsaver_lstm.pth'))
    
    # Save the scaler for inference
    import joblib
    joblib.dump(scaler, os.path.join(os.path.dirname(__file__), 'scaler.pkl'))
    print("Model and scaler saved.")

if __name__ == "__main__":
    train_model()
