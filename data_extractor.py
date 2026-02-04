import pandas as pd
import numpy as np
from pathlib import Path


class DataExtractor:
    """
    A class to extract and process data from CSV/Excel files.
    
    Features:
    - Customizable data path
    - Date column as index
    - Multiple methods for handling missing data
    """
    
    def __init__(self, data_path):
        """
        Initialize the DataExtractor with a data path.
        
        Parameters:
        -----------
        data_path : str or Path
            Path to the data file (CSV or Excel)
        """
        self.data_path = Path(data_path)
        self.df = None
        self.original_df = None
        
    def load_data(self):
        """
        Load data from the specified path.
        Assumes:
        - Column A contains dates (will be set as index)
        - Row 1 contains unique IDs for each column (will be column headers)
        
        Returns:
        --------
        pd.DataFrame : Loaded dataframe
        """
        file_extension = self.data_path.suffix.lower()
        
        try:
            if file_extension == '.csv':
                self.df = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
            elif file_extension in ['.xlsx', '.xls']:
                self.df = pd.read_excel(self.data_path, index_col=0, parse_dates=True)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Store original copy
            self.original_df = self.df.copy()
            
            print(f"Data loaded successfully from {self.data_path}")
            print(f"Shape: {self.df.shape}")
            print(f"Date range: {self.df.index.min()} to {self.df.index.max()}")
            print(f"Missing values: {self.df.isna().sum().sum()}")
            
            return self.df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def handle_missing_data(self, method='drop', **kwargs):
        """
        Handle missing data using various methods.
        
        Parameters:
        -----------
        method : str
            Method to handle missing data. Options:
            - 'drop': Remove rows/columns with NA values
            - 'ffill': Forward fill (propagate last valid observation forward)
            - 'bfill': Backward fill (propagate next valid observation backward)
            - 'mean': Fill with column mean
            - 'median': Fill with column median
            - 'interpolate': Interpolate missing values
        **kwargs : dict
            Additional arguments for specific methods:
            - axis: For 'drop' method (0 for rows, 1 for columns, default=0)
            - how: For 'drop' method ('any' or 'all', default='any')
            - limit: For 'ffill'/'bfill' methods
            - method_type: For 'interpolate' method ('linear', 'time', 'polynomial', etc.)
            - order: For polynomial interpolation
        
        Returns:
        --------
        pd.DataFrame : DataFrame with handled missing data
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        print(f"\nHandling missing data using method: {method}")
        print(f"Missing values before: {self.df.isna().sum().sum()}")
        
        if method == 'drop':
            axis = kwargs.get('axis', 0)
            how = kwargs.get('how', 'any')
            self.df = self.df.dropna(axis=axis, how=how)
            
        elif method == 'ffill':
            limit = kwargs.get('limit', None)
            self.df = self.df.fillna(method='ffill', limit=limit)
            
        elif method == 'bfill':
            limit = kwargs.get('limit', None)
            self.df = self.df.fillna(method='bfill', limit=limit)
            
        elif method == 'mean':
            self.df = self.df.fillna(self.df.mean())
            
        elif method == 'median':
            self.df = self.df.fillna(self.df.median())
            
        elif method == 'interpolate':
            method_type = kwargs.get('method_type', 'linear')
            order = kwargs.get('order', None)
            self.df = self.df.interpolate(method=method_type, order=order)
            
        else:
            raise ValueError(f"Unknown method: {method}. Choose from: 'drop', 'ffill', 'bfill', 'mean', 'median', 'interpolate'")
        
        print(f"Missing values after: {self.df.isna().sum().sum()}")
        
        return self.df
    
    def get_data(self):
        """
        Get the current processed dataframe.
        
        Returns:
        --------
        pd.DataFrame : Current dataframe
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        return self.df
    
    def get_original_data(self):
        """
        Get the original unmodified dataframe.
        
        Returns:
        --------
        pd.DataFrame : Original dataframe
        """
        if self.original_df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        return self.original_df
    
    def reset_to_original(self):
        """
        Reset the dataframe to its original state.
        """
        if self.original_df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        self.df = self.original_df.copy()
        print("Data reset to original state.")
        return self.df
    
    def save_data(self, output_path, **kwargs):
        """
        Save the processed data to a file.
        
        Parameters:
        -----------
        output_path : str or Path
            Path to save the file
        **kwargs : dict
            Additional arguments for pandas to_csv or to_excel
        """
        if self.df is None:
            raise ValueError("No data to save. Call load_data() first.")
        
        output_path = Path(output_path)
        file_extension = output_path.suffix.lower()
        
        try:
            if file_extension == '.csv':
                self.df.to_csv(output_path, **kwargs)
            elif file_extension in ['.xlsx', '.xls']:
                self.df.to_excel(output_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            print(f"Data saved successfully to {output_path}")
            
        except Exception as e:
            print(f"Error saving data: {e}")
            raise
    
    def get_summary(self):
        """
        Get a summary of the current dataframe.
        
        Returns:
        --------
        dict : Summary statistics
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        summary = {
            'shape': self.df.shape,
            'date_range': (self.df.index.min(), self.df.index.max()),
            'columns': list(self.df.columns),
            'missing_values': self.df.isna().sum().to_dict(),
            'total_missing': self.df.isna().sum().sum(),
            'data_types': self.df.dtypes.to_dict()
        }
        
        return summary
