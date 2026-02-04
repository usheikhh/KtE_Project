"""
Configuration file for energy market forecasting framework.
Defines targets, features, and model parameters for gas, electricity, and carbon markets.
"""

# ============================================================
# TARGET DEFINITIONS
# ============================================================

TARGETS = {
    'gas': {
        'name': 'Natural Gas (TTF)',
        'column': 'BM_TTF_M1_CLOSE_EUR_MWH',
        'alternative': 'IM_SPOT_GAS_TTF_DA_PRICE_EUR_MWH',
        'unit': 'EUR/MWh'
    },
    'power': {
        'name': 'Electricity (Germany)',
        'column': 'BM_GERMANY_POWER_M1_CLOSE_EUR_MWH',
        'alternative': 'BM_GERMANY_POWER_CAL1_PRICE_EUR_MWH',
        'unit': 'EUR/MWh'
    },
    'carbon': {
        'name': 'Carbon (EUA)',
        'column': 'BM_EUA_CO2_CAL1_PRICE_EUR_TON',
        'alternative': None,
        'unit': 'EUR/ton'
    }
}

# ============================================================
# SHARED FEATURE BLOCKS
# ============================================================

FEATURE_BLOCKS = {
    'demand': {
        'description': 'Demand indicators (seasonal)',
        'features': [
            # Temperature (actual column names from data)
            'MF_TEMPERATURE_GERMANY_ACTUAL_C',
            'MF_TEMPERATURE_FRANCE_ACTUAL_C',
            'MF_TEMPERATURE_NORMAL_GERMANY_NORMAL_C',
            'MF_TEMPERATURE_NORMAL_FRANCE_NORMAL_C',
            
            # Load (actual column names from data)
            'MF_POWER_LOAD_GERMANY_ACTUAL_LOAD_GW',
            'MF_POWER_LOAD_FRANCE_ACTUAL_LOAD_GW',
            
            # Gas consumption (actual column names from data)
            'MF_GAS_CONSUMPTION_GERMANY_ACTUAL_GWH_D',
            'MF_GAS_CONSUMPTION_FRANCE_ACTUAL_GWH_D',
            'MF_GAS_CONSUMPTION_GERMANY_AVG_2Y_GWH_D',
            'MF_GAS_CONSUMPTION_FRANCE_AVG_2Y_GWH_D',
        ],
        'anomaly_pairs': [
            ('MF_TEMPERATURE_GERMANY_ACTUAL_C', 'MF_TEMPERATURE_NORMAL_GERMANY_NORMAL_C'),
            ('MF_TEMPERATURE_FRANCE_ACTUAL_C', 'MF_TEMPERATURE_NORMAL_FRANCE_NORMAL_C'),
            ('MF_GAS_CONSUMPTION_GERMANY_ACTUAL_GWH_D', 'MF_GAS_CONSUMPTION_GERMANY_AVG_2Y_GWH_D'),
            ('MF_GAS_CONSUMPTION_FRANCE_ACTUAL_GWH_D', 'MF_GAS_CONSUMPTION_FRANCE_AVG_2Y_GWH_D'),
        ]
    },
    
    'supply': {
        'description': 'Supply indicators',
        'features': [
            # Outages and flows (actual column names from data)
            'MF_NORWAY_GAS_OUTAGES_ACTUAL_VALUE_GWH_D',
            'MF_NORWAY_GAS_OUTAGES_SCHEDULED_M1_VALUE_GWH_D',
            'MF_RUSSIAN_PIPELINE_FLOW_ACTUAL_FLOW_GWH_D',
            'MF_LNG_EUROPE_FLOW_ACTUAL_FLOW_GWH_D',
            'MF_NORWAY_GAS_IMPORT_ACTUAL_VALUE_GWH_D',
            
            # Storage (actual column names from data)
            'MF_EU_STORAGE_STORAGE_ACTUAL_STORAGE_TWH',
            
            # Generation (actual column names from data)
            'MF_RENEWABLES_GENERATION_GERMANY_WIND_GENERATION_GW',
            'MF_RENEWABLES_GENERATION_GERMANY_SOLAR_GENERATION_GW',
            'MF_NUCLEAR_GENERATION_FRANCE_GENERATION_GW',
            'MF_LOAD_FACTOR_GERMANY_WIND_LOAD_FACTOR_PCT',
            'MF_LOAD_FACTOR_GERMANY_PV_LOAD_FACTOR_PCT',
        ]
    },
    
    'market_positioning': {
        'description': 'Market positioning and risk indicators',
        'features': [
            # COT positioning (actual column names from data)
            'MF_COT_TTF_HEDGE_FUNDS_NET_MWH',
            'MF_COT_TTF_TOTAL_NET_MWH',
            'MF_COT_EUA_HEDGE_FUNDS_NET_TON',
            'MF_COT_EUA_TOTAL_NET_TON',
            
            # Options and volatility (actual column names from data)
            'IM_TTF_OPTIONS_TTF_IMPLIED_VOL_PCT',
            'IM_TTF_OPTIONS_TTF_PREMIUM_EUR_MWH',
            
            # Fuel switch (actual column names from data)
            'IM_COAL_GAS_SWITCH_M1_SUPPORT_EUR_MWH',
            'IM_COAL_GAS_SWITCH_M1_RESISTANCE_EUR_MWH',
            'IM_COAL_GAS_SWITCH_CAL1_SUPPORT_EUR_MWH',
            'IM_COAL_GAS_SWITCH_CAL1_RESISTANCE_EUR_MWH',
        ]
    },
    
    'cross_commodity': {
        'description': 'Cross-commodity price indicators',
        'features': [
            # Coal (actual column names from data)
            'IM_COAL_CAL1_PRICE_USD_TON',
            
            # Oil (actual column names from data)
            'IM_BRENT_M1_PRICE_USD_BBL',
            
            # International gas (actual column names from data)
            'IM_JKM_LNG_M1_PRICE_USD_MMBTU',
            'IM_HENRY_HUB_SPOT_PRICE_USD_MMBTU',
        ]
    }
}

# ============================================================
# TARGET-SPECIFIC FEATURES
# ============================================================

TARGET_SPECIFIC_FEATURES = {
    'gas': {
        'description': 'Gas-specific must-have drivers',
        'features': [
            # Storage, flows, outages already in shared blocks
            
            # International benchmarks (actual column names)
            'IM_JKM_LNG_M1_PRICE_USD_MMBTU',
            'IM_HENRY_HUB_SPOT_PRICE_USD_MMBTU',
            
            # Oil linkage (actual column names)
            'IM_BRENT_M1_PRICE_USD_BBL',
            
            # Coal-gas switch (actual column names)
            'IM_COAL_GAS_SWITCH_M1_SUPPORT_EUR_MWH',
            'IM_COAL_GAS_SWITCH_CAL1_SUPPORT_EUR_MWH',
            
            # Volatility (actual column names)
            'IM_TTF_OPTIONS_TTF_IMPLIED_VOL_PCT',
            'IM_TTF_OPTIONS_TTF_PREMIUM_EUR_MWH',
        ]
    },
    
    'power': {
        'description': 'Power-specific must-have drivers',
        'features': [
            # Gas price (key input) - actual column names
            'BM_TTF_M1_CLOSE_EUR_MWH',
            'IM_SPOT_GAS_TTF_DA_PRICE_EUR_MWH',
            
            # Carbon price (actual column names)
            'BM_EUA_CO2_CAL1_PRICE_EUR_TON',
            
            # Coal price (actual column names)
            'IM_COAL_CAL1_PRICE_USD_TON',
            
            # Load and generation (actual column names)
            'MF_POWER_LOAD_GERMANY_ACTUAL_LOAD_GW',
            'MF_RENEWABLES_GENERATION_GERMANY_WIND_GENERATION_GW',
            'MF_RENEWABLES_GENERATION_GERMANY_SOLAR_GENERATION_GW',
            'MF_NUCLEAR_GENERATION_FRANCE_GENERATION_GW',
            
            # Transfer capacity (actual column names)
            'MF_TRANSFER_CAPACITY_IMPORT_FRANCE_CAPACITY_GW',
            'MF_TRANSFER_CAPACITY_IMPORT_SCANDINAVIA_CAPACITY_GW',
        ]
    },
    
    'carbon': {
        'description': 'Carbon-specific must-have drivers',
        'features': [
            # Power and gas (generation mix proxy) - actual column names
            'BM_GERMANY_POWER_M1_CLOSE_EUR_MWH',
            'BM_TTF_M1_CLOSE_EUR_MWH',
            
            # Coal-gas switch (switching economics) - actual column names
            'IM_COAL_GAS_SWITCH_M1_SUPPORT_EUR_MWH',
            'IM_COAL_GAS_SWITCH_CAL1_SUPPORT_EUR_MWH',
            'IM_COAL_CAL1_PRICE_USD_TON',
            
            # EUA positioning (actual column names)
            'MF_COT_EUA_HEDGE_FUNDS_NET_TON',
            'MF_COT_EUA_TOTAL_NET_TON',
            
            # Load and renewables (emissions intensity proxy) - actual column names
            'MF_POWER_LOAD_GERMANY_ACTUAL_LOAD_GW',
            'MF_RENEWABLES_GENERATION_GERMANY_WIND_GENERATION_GW',
        ]
    }
}

# ============================================================
# MODEL PARAMETERS
# ============================================================

MODEL_PARAMS = {
    'elastic_net': {
        'alpha': 0.1,  # Overall regularization strength
        'l1_ratio': 0.5,  # Balance between L1 and L2 (0.5 = equal mix)
        'max_iter': 5000,
        'cv_folds': 5,  # For cross-validation to select alpha
    },
    
    'lags': {
        'target_lags': [1, 2, 3, 5, 10],  # Lags of target return
        'feature_lags': [1, 2, 5],  # Lags of key drivers
    },
    
    'validation': {
        'method': 'walk_forward',  # or 'expanding'
        'initial_train_size': 252,  # ~1 year of daily data
        'step_size': 21,  # ~1 month
        'horizon': 1,  # 1-step ahead forecast
    }
}

# ============================================================
# REGIME PARAMETERS (FOR FUTURE USE)
# ============================================================

REGIME_PARAMS = {
    'n_regimes': 2,  # Start with 2 (normal vs stress)
    'regime_type': 'hard',  # 'hard' or 'soft'
    'regime_features': [
        'BM_TTF_M1_CLOSE_EUR_MWH',
        'IM_EU_GAS_STORAGE_PCT',
        'IM_TTF_IMPLIED_VOL_30D_PCT',
    ],
    'enabled': False,  # Set to True to enable regime models
}

# ============================================================
# DATA PROCESSING PARAMETERS
# ============================================================

DATA_PARAMS = {
    'missing_data_method': 'interpolate',  # For feature engineering
    'outlier_threshold': 5,  # Standard deviations for outlier detection
    'scaling_method': 'standard',  # 'standard' or 'robust'
}
