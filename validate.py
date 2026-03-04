import torch
import numpy as np
import os
import sys

# Add parent dir to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib
from data.preprocessing import preprocess_data
from model.lstm import SmartSaverLSTM

def calculate_mape(y_true, y_pred):
    # Avoid division by zero
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100

def validate_model():
    print("Pre-processing hidden month data for validation...")
    df, scaler_from_data = preprocess_data('2024-01-01', '2024-01-31')
    
    # Load trained model and scaler
    model_path = os.path.join(os.path.dirname(__file__), 'smartsaver_lstm.pth')
    scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("Model or scaler not found. Please run train.py first.")
        return
        
    scaler = joblib.load(scaler_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SmartSaverLSTM(input_size=9, output_steps=96, output_features=2)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    data_values = df.values
    
    # Create evaluation sequences
    from model.train import create_sequences
    X, y = create_sequences(data_values)
    
    # Predict
    with torch.no_grad():
        X_tensor = torch.tensor(X).to(device)
        predictions = model(X_tensor).cpu().numpy()
        
    # Here predictions are shape (samples, 96, 2)
    # y is shape (samples, 96, 2)
    # The 2 features are demand and price respectively.
    # In preprocessing, demand and price are the FIRST TWO features transformed by the scaler!
    
    # Let's un-scale them to get real values for MAPE calculation
    # Scaler expects 5 features: ['demand_mw', 'price_rs_per_kwh', 'temperature_c', 'cloud_cover_pct', 'humidity_pct']
    # We pad with zeros for the remaining 3 columns to inverse transform
    
    def inverse_transform_targets(scaled_targets):
        reshaped = scaled_targets.reshape(-1, 2)
        padded = np.zeros((reshaped.shape[0], 5))
        padded[:, :2] = reshaped
        unscaled = scaler.inverse_transform(padded)
        return unscaled[:, :2].reshape(scaled_targets.shape[0], scaled_targets.shape[1], 2)
        
    y_true_unscaled = inverse_transform_targets(y)
    y_pred_unscaled = inverse_transform_targets(predictions)
    
    demand_mape = calculate_mape(y_true_unscaled[..., 0], y_pred_unscaled[..., 0])
    price_mape = calculate_mape(y_true_unscaled[..., 1], y_pred_unscaled[..., 1])
    
    print(f"Validation Results on Hidden Month:")
    print(f"Demand MAPE: {demand_mape:.2f}%")
    print(f"Price MAPE: {price_mape:.2f}%")
    
    if demand_mape < 15.0 and price_mape < 15.0:
        print("Success: Future Vision is adequately trained for agentic inference.")
    else:
        print("Warning: MAPE is high, consider training for more epochs or adding complexity.")

if __name__ == "__main__":
    validate_model()
