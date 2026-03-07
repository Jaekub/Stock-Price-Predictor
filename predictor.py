import sys
import json
import pandas as pd
import numpy as np
import xgboost as xgb

def create_features(df):
    df['return'] = df['close'].pct_change()
    df['sma_5'] = df['close'].rolling(5).mean()
    df['volatility_5'] = df['close'].rolling(5).std()
    df['momentum'] = df['close'] - df['close'].shift(5)

    for i in range(1, 6):
        df[f'lag_{i}'] = df['close'].shift(i)

    df.dropna(inplace=True)
    return df

def train_and_predict(data, days):

    df = pd.DataFrame(data)
    df = create_features(df)

    feature_cols = [
        'return','sma_5','volatility_5','momentum',
        'lag_1','lag_2','lag_3','lag_4','lag_5'
    ]

    X = df[feature_cols]
    y = df['close']

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8
    )

    model.fit(X, y)

    predictions = []
    last_row = df.iloc[-1:].copy()

    for _ in range(days):

        pred = model.predict(last_row[feature_cols])[0]
        predictions.append(float(round(pred, 2)))

        new_row = last_row.copy()
        new_row['close'] = pred

        new_row['return'] = (pred - last_row['close'].values[0]) / last_row['close'].values[0]
        new_row['sma_5'] = np.mean([pred] + predictions[-4:])
        new_row['volatility_5'] = np.std([pred] + predictions[-4:])
        new_row['momentum'] = pred - last_row['lag_5'].values[0]

        for i in range(5, 1, -1):
            new_row[f'lag_{i}'] = last_row[f'lag_{i-1}'].values[0]

        new_row['lag_1'] = pred

        last_row = new_row.copy()

    trend = "bullish" if predictions[-1] > predictions[0] else "bearish"

    return predictions, trend


if __name__ == "__main__":
    days = int(sys.argv[1])
    stock_data = json.load(sys.stdin)

    predictions, trend = train_and_predict(stock_data, days)

    print(json.dumps({
        "days": days,
        "predictions": predictions,
        "trend": trend
    }))
