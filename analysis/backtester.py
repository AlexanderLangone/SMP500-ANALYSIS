
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class BackTester:
    def __init__(self, window_size=252*5):  # 5-year window
        self.window_size = window_size
        
    def rolling_train_test_split(self, data):
        """Implement rolling window splitting for time series"""
        for i in range(self.window_size, len(data), 252):  # Step by 1 year
            train = data.iloc[i-self.window_size:i]
            test = data.iloc[i:i+252]  # Test on next year
            yield train, test
            
    def evaluate_model(self, model, data, target_col='Close'):
        """Evaluate model with proper time series cross-validation"""
        predictions = []
        actuals = []
        
        for train, test in self.rolling_train_test_split(data):
            # Fit model on training data
            X_train = np.arange(len(train)) / 252  # Convert to years
            model.fit(X_train, train[target_col])
            
            # Predict on test data
            X_test = (np.arange(len(test)) + len(train)) / 252
            pred = model.predict(X_test)
            
            predictions.extend(pred)
            actuals.extend(test[target_col])
            
        return {
            'RMSE': np.sqrt(mean_squared_error(actuals, predictions)),
            'MAE': mean_absolute_error(actuals, predictions),
            'R2': r2_score(actuals, predictions)
        }
        
    def classify_regime(self, data, lookback=252):
        """Classify market regime using only past data"""
        df = data.copy()
        df['rolling_mean'] = df['Ratio'].rolling(lookback).mean()
        df['rolling_std'] = df['Ratio'].rolling(lookback).std()
        
        df['Regime'] = 0  # Neutral
        # Strongly overvalued
        df.loc[df['Ratio'] > df['rolling_mean'] + 2*df['rolling_std'], 'Regime'] = 1
        # Mildly overvalued  
        df.loc[(df['Ratio'] > df['rolling_mean'] + df['rolling_std']) & 
               (df['Ratio'] <= df['rolling_mean'] + 2*df['rolling_std']), 'Regime'] = 0.5
        # Mildly undervalued
        df.loc[(df['Ratio'] < df['rolling_mean'] - df['rolling_std']) &
               (df['Ratio'] >= df['rolling_mean'] - 2*df['rolling_std']), 'Regime'] = -0.5
        # Strongly undervalued
        df.loc[df['Ratio'] < df['rolling_mean'] - 2*df['rolling_std'], 'Regime'] = -1
        
        return df['Regime']
