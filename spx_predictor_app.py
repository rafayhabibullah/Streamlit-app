# spx_predictor_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import ta
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# ----- CONFIGURATION ----- #
TIMEFRAME = '15m'
LOOKBACK_CANDLES = 48 * 4  # 192 candles = 48 hours
PREDICT_HORIZON = 1

# ----- LSTM MODEL ----- #
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), 64)
        c0 = torch.zeros(1, x.size(0), 64)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# ----- FAKE SPX DATA LOADER (simulated for demo) ----- #
def load_fake_spx_data():
    # Simulate price movement like SPX over 48 hours
    now = datetime.datetime.now()
    dates = [now - datetime.timedelta(minutes=15 * i) for i in range(LOOKBACK_CANDLES)][::-1]
    prices = np.cumsum(np.random.normal(0, 0.5, size=LOOKBACK_CANDLES)) + 5000
    df = pd.DataFrame({
        'datetime': dates,
        'close': prices,
        'open': prices - np.random.normal(0, 0.5, size=LOOKBACK_CANDLES),
        'high': prices + np.random.normal(0, 1.0, size=LOOKBACK_CANDLES),
        'low': prices - np.random.normal(0, 1.0, size=LOOKBACK_CANDLES),
        'volume': np.random.uniform(1000, 2000, size=LOOKBACK_CANDLES)
    })
    return df

# ----- FEATURE ENGINEERING ----- #
def add_technical_indicators(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['macd'] = ta.trend.MACD(df['close']).macd()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    df = df.dropna()
    return df

# ----- PREPARE DATA FOR LSTM ----- #
def prepare_data(df):
    features = ['close', 'rsi', 'macd', 'atr']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])

    X, y = [], []
    for i in range(len(scaled_data) - 20 - PREDICT_HORIZON):
        X.append(scaled_data[i:i+20])
        target = 1 if scaled_data[i+20+PREDICT_HORIZON-1][0] > scaled_data[i+20-1][0] else 0
        y.append(target)

    return torch.tensor(X).float(), torch.tensor(y).long(), scaler

# ----- TRAIN MODEL (simple, fast) ----- #
def train_model(X, y):
    model = LSTMModel(input_size=X.shape[2], hidden_size=64, num_layers=1, output_size=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    return model

# ----- PREDICT NEXT MOVE ----- #
def predict_next_move(model, recent_data):
    model.eval()
    with torch.no_grad():
        output = model(recent_data.unsqueeze(0))
        probs = torch.softmax(output, dim=1).numpy()[0]
        return probs

# ----- STREAMLIT UI ----- #
st.title("📈 SPX/USDT 15m AI Market Predictor")

data = load_fake_spx_data()
data = add_technical_indicators(data)
X, y, scaler = prepare_data(data)
model = train_model(X, y)

recent_sequence = X[-1]
prediction_probs = predict_next_move(model, recent_sequence)
classes = ['DOWN', 'UP']

st.subheader("Latest Prediction")
pred_class = classes[np.argmax(prediction_probs)]
confidence = np.max(prediction_probs)

st.metric(label="Prediction", value=pred_class, delta=f"Confidence: {confidence:.2%}")

# Plot chart
st.subheader("Price Chart (last 48h)")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(data['datetime'], data['close'], label='Close Price')
ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

st.caption("Note: Model trained on simulated SPX-style data for demonstration.")
