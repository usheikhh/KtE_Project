"""
Target transformation utilities for energy market forecasting.
Handles log-return transformations and inverse transformations.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
import config


class TargetTransformer:
    """
    Transform targets to log-returns and back to prices.
    
    y_t = Δlog(P_t) = log(P_t) - log(P_{t-1})
    
    This reduces shared trend problems and makes coefficients interpretable
    as "impact on percent change".
    """
    
    def __init__(self, target_name: str):
        """
        Initialize transformer for a specific target.
        
        Parameters:
        -----------
        target_name : str
            Target name ('gas', 'power', or 'carbon')
        """
        self.target_name = target_name
        self.target_config = config.TARGETS[target_name]
        self.price_column = self.target_config['column']
        self.last_price = None
        
    def transform_to_returns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Transform price series to log-returns.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe with price column
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.Series]
            (dataframe with returns, original prices)
        """
        if self.price_column not in df.columns:
            raise ValueError(f"Price column {self.price_column} not found in dataframe")
        
        prices = df[self.price_column].copy()
        
        # Calculate log returns
        log_returns = np.log(prices / prices.shift(1))
        
        # Store for inverse transform
        self.last_price = prices
        
        # Create new dataframe with returns
        df_returns = df.copy()
        df_returns[f'{self.target_name}_return'] = log_returns
        
        print(f"\n{'='*60}")
        print(f"Target Transformation: {self.target_config['name']}")
        print(f"{'='*60}")
        print(f"Original price column: {self.price_column}")
        print(f"Price range: {prices.min():.2f} - {prices.max():.2f} {self.target_config['unit']}")
        print(f"Return column: {self.target_name}_return")
        print(f"Return mean: {log_returns.mean():.6f}")
        print(f"Return std: {log_returns.std():.6f}")
        print(f"Non-null observations: {log_returns.notna().sum()}")
        
        return df_returns, prices
    
    def inverse_transform(self, returns: pd.Series, 
                         initial_price: Optional[float] = None,
                         reference_prices: Optional[pd.Series] = None) -> pd.Series:
        """
        Convert log-returns back to prices.
        
        P_t = P_{t-1} * exp(y_t)
        
        Parameters:
        -----------
        returns : pd.Series
            Log-returns
        initial_price : float, optional
            Starting price for the series
        reference_prices : pd.Series, optional
            Reference price series (for aligning forecasts)
            
        Returns:
        --------
        pd.Series : Price series
        """
        if initial_price is None:
            if reference_prices is not None:
                initial_price = reference_prices.iloc[0]
            elif self.last_price is not None:
                initial_price = self.last_price.iloc[0]
            else:
                raise ValueError("Need either initial_price or reference_prices")
        
        # Reconstruct prices from returns
        # P_t = P_0 * exp(sum of returns from 0 to t)
        cumulative_returns = returns.fillna(0).cumsum()
        prices = initial_price * np.exp(cumulative_returns)
        
        return prices
    
    def forecast_to_price(self, forecast_return: float, 
                         last_known_price: float) -> float:
        """
        Convert a single return forecast to price forecast.
        
        Parameters:
        -----------
        forecast_return : float
            Forecasted log-return
        last_known_price : float
            Last known price
            
        Returns:
        --------
        float : Forecasted price
        """
        return last_known_price * np.exp(forecast_return)
    
    def get_target_column(self, df: pd.DataFrame, as_returns: bool = True) -> pd.Series:
        """
        Get the target column (either as returns or prices).
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe with target data
        as_returns : bool
            If True, return log-returns; if False, return prices
            
        Returns:
        --------
        pd.Series : Target series
        """
        if as_returns:
            return_col = f'{self.target_name}_return'
            if return_col in df.columns:
                return df[return_col]
            else:
                # Transform on the fly
                df_transformed, _ = self.transform_to_returns(df)
                return df_transformed[return_col]
        else:
            if self.price_column in df.columns:
                return df[self.price_column]
            else:
                raise ValueError(f"Price column {self.price_column} not found")


def prepare_all_targets(df: pd.DataFrame) -> Dict[str, Tuple[pd.DataFrame, TargetTransformer]]:
    """
    Prepare all three targets with transformations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw data
        
    Returns:
    --------
    Dict : Dictionary with target data and transformers
    """
    results = {}
    
    for target_name in ['gas', 'power', 'carbon']:
        print(f"\n{'='*70}")
        print(f"Preparing target: {target_name.upper()}")
        print(f"{'='*70}")
        
        transformer = TargetTransformer(target_name)
        df_transformed, original_prices = transformer.transform_to_returns(df)
        
        results[target_name] = {
            'transformer': transformer,
            'data': df_transformed,
            'prices': original_prices,
            'return_column': f'{target_name}_return'
        }
    
    return results
