"""
Model and validation framework for energy market forecasting.
Implements Elastic Net with lags and walk-forward validation.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Optional
import config


class EnergyMarketModel:
    """
    Elastic Net regression model with lag features for energy markets.
    
    y_t = α + Σ φ_i * y_{t-i} + X_t^T * β + ε_t
    """
    
    def __init__(self, target_name: str, use_cv: bool = True):
        """
        Initialize model for a specific target.
        
        Parameters:
        -----------
        target_name : str
            Target name ('gas', 'power', or 'carbon')
        use_cv : bool
            Whether to use cross-validation for hyperparameter selection
        """
        self.target_name = target_name
        self.use_cv = use_cv
        
        # Initialize model
        if use_cv:
            self.model = ElasticNetCV(
                l1_ratio=config.MODEL_PARAMS['elastic_net']['l1_ratio'],
                cv=config.MODEL_PARAMS['elastic_net']['cv_folds'],
                max_iter=config.MODEL_PARAMS['elastic_net']['max_iter'],
                random_state=42
            )
        else:
            self.model = ElasticNet(
                alpha=config.MODEL_PARAMS['elastic_net']['alpha'],
                l1_ratio=config.MODEL_PARAMS['elastic_net']['l1_ratio'],
                max_iter=config.MODEL_PARAMS['elastic_net']['max_iter'],
                random_state=42
            )
        
        # Scaler for features
        self.scaler = StandardScaler()
        
        # Store fitted state
        self.is_fitted = False
        self.feature_names = None
        self.coef_df = None
        
    def add_target_lags(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Add lagged values of the target to features.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target series
            
        Returns:
        --------
        pd.DataFrame : Features with target lags
        """
        X_with_lags = X.copy()
        
        for lag in config.MODEL_PARAMS['lags']['target_lags']:
            lag_name = f'target_lag_{lag}'
            X_with_lags[lag_name] = y.shift(lag)
        
        return X_with_lags
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'EnergyMarketModel':
        """
        Fit the model.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target (log-returns)
            
        Returns:
        --------
        self : Fitted model
        """
        # Add target lags
        X_with_lags = self.add_target_lags(X, y)
        
        # Remove rows with NaN (from lags)
        valid_idx = X_with_lags.notna().all(axis=1) & y.notna()
        X_clean = X_with_lags[valid_idx]
        y_clean = y[valid_idx]
        
        # Store feature names
        self.feature_names = X_clean.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Fit model
        self.model.fit(X_scaled, y_clean)
        self.is_fitted = True
        
        # Store coefficients
        self._extract_coefficients()
        
        return self
    
    def predict(self, X: pd.DataFrame, y_for_lags: Optional[pd.Series] = None) -> pd.Series:
        """
        Make predictions.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y_for_lags : pd.Series, optional
            Target series for creating lags (needed for multi-step forecast)
            
        Returns:
        --------
        pd.Series : Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Add target lags
        if y_for_lags is not None:
            X_with_lags = self.add_target_lags(X, y_for_lags)
        else:
            # Assume lags already in X
            X_with_lags = X.copy()
        
        # Align with training features
        X_aligned = X_with_lags[self.feature_names]
        
        # Remove rows with NaN
        valid_idx = X_aligned.notna().all(axis=1)
        X_clean = X_aligned[valid_idx]
        
        # Scale and predict
        X_scaled = self.scaler.transform(X_clean)
        predictions = self.model.predict(X_scaled)
        
        # Return as series with proper index
        return pd.Series(predictions, index=X_clean.index)
    
    def _extract_coefficients(self):
        """Extract and sort coefficients by absolute value."""
        if not self.is_fitted:
            return
        
        coef_dict = {
            'feature': self.feature_names,
            'coefficient': self.model.coef_,
            'abs_coefficient': np.abs(self.model.coef_)
        }
        
        self.coef_df = pd.DataFrame(coef_dict).sort_values(
            'abs_coefficient', ascending=False
        )
        
        # Add intercept
        intercept_row = pd.DataFrame({
            'feature': ['intercept'],
            'coefficient': [self.model.intercept_],
            'abs_coefficient': [np.abs(self.model.intercept_)]
        })
        
        self.coef_df = pd.concat([intercept_row, self.coef_df], ignore_index=True)
    
    def get_top_features(self, n: int = 20) -> pd.DataFrame:
        """
        Get top N features by absolute coefficient value.
        
        Parameters:
        -----------
        n : int
            Number of top features to return
            
        Returns:
        --------
        pd.DataFrame : Top features with coefficients
        """
        if self.coef_df is None:
            raise ValueError("Model not fitted or coefficients not extracted")
        
        return self.coef_df.head(n)
    
    def get_model_info(self) -> Dict:
        """Get model information and hyperparameters."""
        info = {
            'target': self.target_name,
            'is_fitted': self.is_fitted,
            'n_features': len(self.feature_names) if self.feature_names else 0,
        }
        
        if self.is_fitted:
            info['alpha'] = self.model.alpha_ if self.use_cv else self.model.alpha
            info['l1_ratio'] = self.model.l1_ratio
            info['n_iter'] = self.model.n_iter_
            info['intercept'] = self.model.intercept_
            info['n_nonzero_coef'] = np.sum(np.abs(self.model.coef_) > 1e-10)
        
        return info


class WalkForwardValidator:
    """
    Walk-forward validation for time series models.
    """
    
    def __init__(self, initial_train_size: int = 252, 
                 step_size: int = 21, horizon: int = 1):
        """
        Initialize validator.
        
        Parameters:
        -----------
        initial_train_size : int
            Initial training window size (in days)
        step_size : int
            Step size for moving forward (in days)
        horizon : int
            Forecast horizon (1 = 1-step ahead)
        """
        self.initial_train_size = initial_train_size
        self.step_size = step_size
        self.horizon = horizon
        self.results = []
        
    def validate(self, model: EnergyMarketModel, 
                X: pd.DataFrame, y: pd.Series,
                expanding: bool = False) -> pd.DataFrame:
        """
        Perform walk-forward validation.
        
        Parameters:
        -----------
        model : EnergyMarketModel
            Model to validate
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target series
        expanding : bool
            If True, use expanding window; if False, use rolling window
            
        Returns:
        --------
        pd.DataFrame : Validation results
        """
        print(f"\n{'='*70}")
        print(f"Walk-Forward Validation: {model.target_name.upper()}")
        print(f"{'='*70}")
        print(f"Initial train size: {self.initial_train_size} days")
        print(f"Step size: {self.step_size} days")
        print(f"Window type: {'Expanding' if expanding else 'Rolling'}")
        
        results = []
        n_samples = len(X)
        
        # Start validation loop
        start_idx = self.initial_train_size
        fold = 0
        
        while start_idx + self.step_size < n_samples:
            fold += 1
            
            # Define train and test windows
            if expanding:
                train_start = 0
            else:
                train_start = max(0, start_idx - self.initial_train_size)
            
            train_end = start_idx
            test_start = start_idx
            test_end = min(start_idx + self.step_size, n_samples)
            
            # Split data
            X_train = X.iloc[train_start:train_end]
            y_train = y.iloc[train_start:train_end]
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test, y_for_lags=y)
            
            # Align predictions and actuals
            common_idx = y_pred.index.intersection(y_test.index)
            if len(common_idx) == 0:
                print(f"Warning: No common indices in fold {fold}, skipping")
                start_idx += self.step_size
                continue
            
            y_pred_aligned = y_pred.loc[common_idx]
            y_test_aligned = y_test.loc[common_idx]
            
            # Calculate metrics
            mse = mean_squared_error(y_test_aligned, y_pred_aligned)
            mae = mean_absolute_error(y_test_aligned, y_pred_aligned)
            r2 = r2_score(y_test_aligned, y_pred_aligned)
            
            # Store results
            results.append({
                'fold': fold,
                'train_start': X.index[train_start],
                'train_end': X.index[train_end-1],
                'test_start': X.index[test_start],
                'test_end': X.index[test_end-1],
                'train_size': train_end - train_start,
                'test_size': len(common_idx),
                'mse': mse,
                'rmse': np.sqrt(mse),
                'mae': mae,
                'r2': r2,
            })
            
            print(f"Fold {fold}: RMSE={np.sqrt(mse):.6f}, MAE={mae:.6f}, R²={r2:.4f}")
            
            # Move forward
            start_idx += self.step_size
        
        self.results = pd.DataFrame(results)
        
        # Print summary
        print(f"\n{'='*70}")
        print("Validation Summary:")
        print(f"Total folds: {len(self.results)}")
        print(f"Average RMSE: {self.results['rmse'].mean():.6f}")
        print(f"Average MAE: {self.results['mae'].mean():.6f}")
        print(f"Average R²: {self.results['r2'].mean():.4f}")
        print(f"{'='*70}")
        
        return self.results
    
    def get_results(self) -> pd.DataFrame:
        """Get validation results."""
        return self.results
