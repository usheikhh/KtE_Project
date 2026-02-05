"""
Feature engineering module for energy market forecasting.
Builds features in blocks: demand, supply, market positioning, cross-commodity.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import config


class FeatureEngineer:
    """
    Build features for energy market forecasting with:
    - Shared feature blocks (demand, supply, market, cross-commodity)
    - Target-specific additions
    - Anomaly creation for seasonality isolation
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with raw data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw data with date index and all columns
        """
        self.df = df.copy()
        self.features = pd.DataFrame(index=df.index)
        self.feature_names = []
        
    def create_anomalies(self, actual_col: str, normal_col: str, 
                        anomaly_name: Optional[str] = None) -> str:
        """
        Create anomaly feature: actual - normal.
        
        Parameters:
        -----------
        actual_col : str
            Column with actual values
        normal_col : str
            Column with normal/average values
        anomaly_name : str, optional
            Name for the anomaly feature
            
        Returns:
        --------
        str : Name of the created anomaly feature
        """
        if anomaly_name is None:
            anomaly_name = f"{actual_col}_ANOMALY"
        
        if actual_col in self.df.columns and normal_col in self.df.columns:
            self.features[anomaly_name] = self.df[actual_col] - self.df[normal_col]
            self.feature_names.append(anomaly_name)
            print(f"Created anomaly: {anomaly_name}")
        else:
            print(f"Warning: Could not create {anomaly_name} - columns missing")
        
        return anomaly_name
    
    def build_demand_block(self) -> pd.DataFrame:
        """
        Build demand block features.
        
        Returns:
        --------
        pd.DataFrame : Demand features
        """
        print("\n" + "="*60)
        print("Building DEMAND block")
        print("="*60)
        
        demand_config = config.FEATURE_BLOCKS['demand']
        
        # Add raw features that exist
        for feat in demand_config['features']:
            if feat in self.df.columns:
                self.features[feat] = self.df[feat]
                self.feature_names.append(feat)
        
        # Create anomalies
        for actual_col, normal_col in demand_config['anomaly_pairs']:
            self.create_anomalies(actual_col, normal_col)
        
        return self.features
    
    def build_supply_block(self) -> pd.DataFrame:
        """
        Build supply block features.
        
        Returns:
        --------
        pd.DataFrame : Supply features
        """
        print("\n" + "="*60)
        print("Building SUPPLY block")
        print("="*60)
        
        supply_config = config.FEATURE_BLOCKS['supply']
        
        # Add raw features that exist
        for feat in supply_config['features']:
            if feat in self.df.columns:
                self.features[feat] = self.df[feat]
                self.feature_names.append(feat)
            else:
                print(f"Warning: Feature not found: {feat}")
        
        # Create storage anomaly (current vs historical average)
        if 'MF_EU_STORAGE_STORAGE_ACTUAL_STORAGE_TWH' in self.df.columns:
            # Rolling 2-year average
            storage_avg = self.df['MF_EU_STORAGE_STORAGE_ACTUAL_STORAGE_TWH'].rolling(
                window=504, min_periods=100
            ).mean()
            self.features['MF_EU_STORAGE_ANOMALY'] = (
                self.df['MF_EU_STORAGE_STORAGE_ACTUAL_STORAGE_TWH'] - storage_avg
            )
            self.feature_names.append('MF_EU_STORAGE_ANOMALY')
        
        return self.features
    
    def build_market_positioning_block(self) -> pd.DataFrame:
        """
        Build market positioning and risk block features.
        
        Returns:
        --------
        pd.DataFrame : Market positioning features
        """
        print("\n" + "="*60)
        print("Building MARKET POSITIONING block")
        print("="*60)
        
        market_config = config.FEATURE_BLOCKS['market_positioning']
        
        # Add raw features that exist
        for feat in market_config['features']:
            if feat in self.df.columns:
                self.features[feat] = self.df[feat]
                self.feature_names.append(feat)
            else:
                print(f"Warning: Feature not found: {feat}")
        
        return self.features
    
    def build_cross_commodity_block(self) -> pd.DataFrame:
        """
        Build cross-commodity price features.
        
        Returns:
        --------
        pd.DataFrame : Cross-commodity features
        """
        print("\n" + "="*60)
        print("Building CROSS-COMMODITY block")
        print("="*60)
        
        cross_config = config.FEATURE_BLOCKS['cross_commodity']
        
        # Add raw features that exist
        for feat in cross_config['features']:
            if feat in self.df.columns:
                self.features[feat] = self.df[feat]
                self.feature_names.append(feat)
            else:
                print(f"Warning: Feature not found: {feat}")
        
        # Add returns for price features (useful for cross-commodity dynamics)
        price_cols = [
            'IM_COAL_CAL1_PRICE_USD_TON',
            'IM_BRENT_M1_PRICE_USD_BBL',
            'IM_JKM_LNG_M1_PRICE_USD_MMBTU',
            'IM_HENRY_HUB_SPOT_PRICE_USD_MMBTU',
        ]
        
        for col in price_cols:
            if col in self.df.columns:
                return_name = f"{col}_RETURN"
                self.features[return_name] = np.log(self.df[col] / self.df[col].shift(1))
                self.feature_names.append(return_name)
        
        return self.features
    
    def add_target_specific_features(self, target: str) -> pd.DataFrame:
        """
        Add target-specific features.
        
        Parameters:
        -----------
        target : str
            Target name ('gas', 'power', or 'carbon')
            
        Returns:
        --------
        pd.DataFrame : Updated features with target-specific additions
        """
        print("\n" + "="*60)
        print(f"Adding TARGET-SPECIFIC features for {target.upper()}")
        print("="*60)
        
        if target not in config.TARGET_SPECIFIC_FEATURES:
            print(f"Warning: No target-specific config for {target}")
            return self.features
        
        target_config = config.TARGET_SPECIFIC_FEATURES[target]
        
        # Add features that aren't already included
        for feat in target_config['features']:
            if feat in self.df.columns and feat not in self.feature_names:
                self.features[feat] = self.df[feat]
                self.feature_names.append(feat)
                print(f"Added: {feat}")
        
        return self.features
    
    def add_lags(self, columns: List[str], lags: List[int]) -> pd.DataFrame:
        """
        Add lagged features.
        
        Parameters:
        -----------
        columns : List[str]
            Columns to create lags for
        lags : List[int]
            Lag periods
            
        Returns:
        --------
        pd.DataFrame : Features with lags
        """
        print(f"\nAdding lags {lags} for {len(columns)} features...")
        
        for col in columns:
            if col in self.features.columns:
                for lag in lags:
                    lag_name = f"{col}_LAG{lag}"
                    self.features[lag_name] = self.features[col].shift(lag)
                    self.feature_names.append(lag_name)
        
        return self.features
    
    def build_all_features(self, target: str, 
                          add_lags_to_features: bool = True) -> pd.DataFrame:
        """
        Build all features for a specific target.
        
        Parameters:
        -----------
        target : str
            Target name ('gas', 'power', or 'carbon')
        add_lags_to_features : bool
            Whether to add lags to key features
            
        Returns:
        --------
        pd.DataFrame : Complete feature set
        """
        print("\n" + "="*70)
        print(f"BUILDING COMPLETE FEATURE SET FOR {target.upper()}")
        print("="*70)
        
        # Build shared blocks
        self.build_demand_block()
        self.build_supply_block()
        self.build_market_positioning_block()
        self.build_cross_commodity_block()
        
        # Add target-specific features
        self.add_target_specific_features(target)
        
        # Add lags to key features
        if add_lags_to_features:
            key_features = self._get_key_features_for_lags(target)
            self.add_lags(key_features, config.MODEL_PARAMS['lags']['feature_lags'])
        
        # Remove any all-NaN columns
        self.features = self.features.dropna(axis=1, how='all')
        
        print("\n" + "="*70)
        print(f"Feature engineering complete!")
        print(f"Total features: {len(self.features.columns)}")
        print(f"Date range: {self.features.index.min()} to {self.features.index.max()}")
        print("="*70)
        
        return self.features
    
    def _get_key_features_for_lags(self, target: str) -> List[str]:
        """
        Get key features to create lags for based on target.
        
        Parameters:
        -----------
        target : str
            Target name
            
        Returns:
        --------
        List[str] : List of feature names
        """
        # Common key features across all targets
        key_features = []
        
        # Add existing features only
        candidates = [
            'MF_TEMPERATURE_GERMANY_ACTUAL_C_ANOMALY',
            'MF_EU_STORAGE_STORAGE_ACTUAL_STORAGE_TWH',
            'MF_POWER_LOAD_GERMANY_ACTUAL_LOAD_GW',
        ]
        
        for feat in candidates:
            if feat in self.features.columns:
                key_features.append(feat)
        
        # Target-specific key features
        if target == 'gas':
            candidates = ['MF_LNG_EUROPE_FLOW_ACTUAL_FLOW_GWH_D', 'IM_TTF_OPTIONS_TTF_IMPLIED_VOL_PCT']
        elif target == 'power':
            candidates = ['BM_TTF_M1_CLOSE_EUR_MWH', 'MF_RENEWABLES_GENERATION_GERMANY_WIND_GENERATION_GW']
        elif target == 'carbon':
            candidates = ['IM_COAL_GAS_SWITCH_M1_SUPPORT_EUR_MWH', 'MF_COT_EUA_HEDGE_FUNDS_NET_TON']
        
        for feat in candidates:
            if feat in self.features.columns:
                key_features.append(feat)
        
        return key_features
    
    def get_features(self) -> pd.DataFrame:
        """Get the current feature dataframe."""
        return self.features
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names
