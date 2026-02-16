#!/usr/bin/env python3
"""
Stock Price Prediction Model
Uses XGBoost with technical indicators for price forecasting
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
try:
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import xgboost as xgb
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call(['pip', 'install', '--break-system-packages', 'scikit-learn', 'xgboost', 'pandas', 'numpy'])
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import xgboost as xgb


class StockPredictor:
    """
    Advanced stock price prediction using XGBoost with technical indicators
    """
    
    def __init__(self, lookback_days=60):
        self.lookback_days = lookback_days
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def calculate_technical_indicators(self, df):
        """
        Calculate comprehensive technical indicators
        
        Args:
            df: DataFrame with columns ['date', 'open', 'high', 'low', 'close', 'volume']
        
        Returns:
            DataFrame with additional technical indicator columns
        """
        data = df.copy()
        
        # Price-based features
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        data['high_low_range'] = (data['high'] - data['low']) / data['close']
        data['open_close_range'] = (data['close'] - data['open']) / data['open']
        
        # Moving Averages
        for period in [5, 10, 20, 50]:
            data[f'sma_{period}'] = data['close'].rolling(window=period).mean()
            data[f'ema_{period}'] = data['close'].ewm(span=period, adjust=False).mean()
            data[f'price_to_sma_{period}'] = data['close'] / data[f'sma_{period}']
        
        # Momentum Indicators
        # RSI (Relative Strength Index)
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        exp1 = data['close'].ewm(span=12, adjust=False).mean()
        exp2 = data['close'].ewm(span=26, adjust=False).mean()
        data['macd'] = exp1 - exp2
        data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        # Bollinger Bands
        data['bb_middle'] = data['close'].rolling(window=20).mean()
        std = data['close'].rolling(window=20).std()
        data['bb_upper'] = data['bb_middle'] + (std * 2)
        data['bb_lower'] = data['bb_middle'] - (std * 2)
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # Volatility
        data['volatility_10'] = data['returns'].rolling(window=10).std()
        data['volatility_30'] = data['returns'].rolling(window=30).std()
        
        # Volume indicators
        data['volume_sma_20'] = data['volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_sma_20']
        
        # Average True Range (ATR)
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        data['atr_14'] = true_range.rolling(window=14).mean()
        
        # Stochastic Oscillator
        low_14 = data['low'].rolling(window=14).min()
        high_14 = data['high'].rolling(window=14).max()
        data['stoch_k'] = 100 * ((data['close'] - low_14) / (high_14 - low_14))
        data['stoch_d'] = data['stoch_k'].rolling(window=3).mean()
        
        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            data[f'roc_{period}'] = ((data['close'] - data['close'].shift(period)) / 
                                     data['close'].shift(period)) * 100
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            data[f'close_lag_{lag}'] = data['close'].shift(lag)
            data[f'volume_lag_{lag}'] = data['volume'].shift(lag)
        
        # Time-based features
        data['day_of_week'] = pd.to_datetime(data['date']).dt.dayofweek
        data['day_of_month'] = pd.to_datetime(data['date']).dt.day
        data['month'] = pd.to_datetime(data['date']).dt.month
        
        return data
    
    def prepare_features(self, df):
        """
        Prepare feature matrix for training/prediction
        """
        # Calculate all indicators
        df_features = self.calculate_technical_indicators(df)
        
        # Drop NaN values (from indicator calculations)
        df_features = df_features.dropna()
        
        # Select feature columns (exclude date, open, high, low, close, volume)
        exclude_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'timestamp']
        feature_cols = [col for col in df_features.columns if col not in exclude_cols]
        
        self.feature_names = feature_cols
        
        return df_features[feature_cols], df_features['close']
    
    def train(self, df, test_size=0.2):
        """
        Train the XGBoost model
        
        Args:
            df: DataFrame with stock data
            test_size: Proportion of data for testing
        
        Returns:
            dict: Training metrics
        """
        print("Preparing features...")
        X, y = self.prepare_features(df)
        
        # Time series split (preserve temporal order)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost model
        print("Training XGBoost model...")
        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False
        )
        
        # Evaluate
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        metrics = {
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_accuracy': self._directional_accuracy(y_train, y_pred_train),
            'test_accuracy': self._directional_accuracy(y_test, y_pred_test)
        }
        
        print(f"\n{'='*60}")
        print("Model Performance:")
        print(f"{'='*60}")
        print(f"Train MAE:  ₹{metrics['train_mae']:.2f}")
        print(f"Test MAE:   ₹{metrics['test_mae']:.2f}")
        print(f"Train RMSE: ₹{metrics['train_rmse']:.2f}")
        print(f"Test RMSE:  ₹{metrics['test_rmse']:.2f}")
        print(f"Train R²:   {metrics['train_r2']:.4f}")
        print(f"Test R²:    {metrics['test_r2']:.4f}")
        print(f"Train Directional Accuracy: {metrics['train_accuracy']:.2f}%")
        print(f"Test Directional Accuracy:  {metrics['test_accuracy']:.2f}%")
        print(f"{'='*60}\n")
        
        return metrics
    
    def _directional_accuracy(self, y_true, y_pred):
        """Calculate percentage of correct directional predictions"""
        y_true_arr = np.array(y_true)
        direction_true = np.sign(y_true_arr[1:] - y_true_arr[:-1])
        direction_pred = np.sign(y_pred[1:] - y_pred[:-1])
        correct = np.sum(direction_true == direction_pred)
        return (correct / len(direction_true)) * 100
    
    def predict_future(self, df, days=30):
        """
        Predict future prices
        
        Args:
            df: Historical data DataFrame
            days: Number of days to predict
        
        Returns:
            dict: Predictions with dates and confidence intervals
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = []
        dates = []
        confidence_intervals = []
        
        # Start from the last date in data
        last_date = pd.to_datetime(df['date'].iloc[-1])
        
        # Use the last window of data for rolling predictions
        current_df = df.copy()
        
        for i in range(days):
            # Prepare features for current state
            X, _ = self.prepare_features(current_df)
            X_scaled = self.scaler.transform(X.tail(1))
            
            # Predict next day
            pred = self.model.predict(X_scaled)[0]
            predictions.append(float(pred))
            
            # Calculate prediction date (skip weekends for stock market)
            next_date = last_date + timedelta(days=1)
            while next_date.weekday() >= 5:  # Skip Saturday (5) and Sunday (6)
                next_date += timedelta(days=1)
            dates.append(next_date.strftime('%Y-%m-%d'))
            last_date = next_date
            
            # Estimate confidence interval (simplified)
            # In production, use proper prediction intervals from model
            std_error = current_df['close'].std() * 0.02  # Approximate
            conf_lower = pred - (1.96 * std_error)
            conf_upper = pred + (1.96 * std_error)
            confidence_intervals.append({
                'lower': float(conf_lower),
                'upper': float(conf_upper)
            })
            
            # Add prediction to dataframe for next iteration
            new_row = pd.DataFrame({
                'date': [next_date.strftime('%Y-%m-%d')],
                'open': [pred],
                'high': [pred * 1.005],
                'low': [pred * 0.995],
                'close': [pred],
                'volume': [current_df['volume'].tail(10).mean()],
                'timestamp': [0]
            })
            current_df = pd.concat([current_df, new_row], ignore_index=True)
        
        return {
            'dates': dates,
            'predictions': predictions,
            'confidence_intervals': confidence_intervals,
            'summary': {
                'mean': float(np.mean(predictions)),
                'max': float(np.max(predictions)),
                'min': float(np.min(predictions)),
                'trend': 'bullish' if predictions[-1] > predictions[0] else 'bearish'
            }
        }
    
    def get_feature_importance(self, top_n=10):
        """Get top N most important features"""
        if self.model is None:
            return None
        
        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance.head(top_n).to_dict('records')


# Example usage
if __name__ == "__main__":
    # Sample data structure (would come from your API)
    sample_data = {
        'date': pd.date_range(start='2024-08-01', periods=150, freq='D'),
        'open': np.random.randn(150).cumsum() + 1500,
        'high': np.random.randn(150).cumsum() + 1510,
        'low': np.random.randn(150).cumsum() + 1490,
        'close': np.random.randn(150).cumsum() + 1500,
        'volume': np.random.randint(1000000, 5000000, 150)
    }
    df_sample = pd.DataFrame(sample_data)
    df_sample['timestamp'] = 0
    
    # Initialize and train
    predictor = StockPredictor(lookback_days=60)
    metrics = predictor.train(df_sample)
    
    # Make predictions
    future_predictions = predictor.predict_future(df_sample, days=30)
    
    print("\nFuture Predictions (next 7 days):")
    for i in range(min(7, len(future_predictions['dates']))):
        print(f"{future_predictions['dates'][i]}: ₹{future_predictions['predictions'][i]:.2f} "
              f"(Range: ₹{future_predictions['confidence_intervals'][i]['lower']:.2f} - "
              f"₹{future_predictions['confidence_intervals'][i]['upper']:.2f})")
    
    print(f"\n30-Day Summary:")
    print(f"  Average: ₹{future_predictions['summary']['mean']:.2f}")
    print(f"  High:    ₹{future_predictions['summary']['max']:.2f}")
    print(f"  Low:     ₹{future_predictions['summary']['min']:.2f}")
    print(f"  Trend:   {future_predictions['summary']['trend']}")
    
    print("\nTop 10 Important Features:")
    for feat in predictor.get_feature_importance(10):
        print(f"  {feat['feature']}: {feat['importance']:.4f}")
