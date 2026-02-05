"""
Example script showing how to use the DataExtractor class.
Adjust the DATA_PATH variable to point to your data file.
"""

from data_extractor import DataExtractor

# ============================================================
# USER CONFIGURATION
# ============================================================
# Adjust this path to your data file
DATA_PATH = "raw_data_2021_2024.csv"

# Choose a method to handle missing data:
# Options: 'drop', 'ffill', 'bfill', 'mean', 'median', 'interpolate'
MISSING_DATA_METHOD = 'ffill'

# Optional: Specify output path to save processed data
OUTPUT_PATH = "processed_data.csv"
# ============================================================


def main():
    """Main function to demonstrate DataExtractor usage."""
    
    # Initialize the extractor
    print("="*60)
    print("Data Extraction Process")
    print("="*60)
    
    extractor = DataExtractor(DATA_PATH)
    
    # Load the data
    df = extractor.load_data()
    
    # Display basic information
    print("\n" + "="*60)
    print("Data Preview:")
    print("="*60)
    print(df.head())
    
    # Handle missing data
    print("\n" + "="*60)
    print("Processing Missing Data")
    print("="*60)
    
    # Example 1: Forward fill
    if MISSING_DATA_METHOD == 'ffill':
        df_processed = extractor.handle_missing_data(method='ffill', limit=3)
    
    # Example 2: Backward fill
    elif MISSING_DATA_METHOD == 'bfill':
        df_processed = extractor.handle_missing_data(method='bfill', limit=3)
    
    # Example 3: Drop rows with any missing values
    elif MISSING_DATA_METHOD == 'drop':
        df_processed = extractor.handle_missing_data(method='drop', axis=0, how='any')
    
    # Example 4: Fill with mean
    elif MISSING_DATA_METHOD == 'mean':
        df_processed = extractor.handle_missing_data(method='mean')
    
    # Example 5: Fill with median
    elif MISSING_DATA_METHOD == 'median':
        df_processed = extractor.handle_missing_data(method='median')
    
    # Example 6: Interpolate (linear by default)
    elif MISSING_DATA_METHOD == 'interpolate':
        df_processed = extractor.handle_missing_data(method='interpolate', method_type='linear')
    
    else:
        print(f"Unknown method: {MISSING_DATA_METHOD}")
        return
    
    # Display processed data preview
    print("\n" + "="*60)
    print("Processed Data Preview:")
    print("="*60)
    print(df_processed.head())
    
    # Get summary
    print("\n" + "="*60)
    print("Data Summary:")
    print("="*60)
    summary = extractor.get_summary()
    print(f"Shape: {summary['shape']}")
    print(f"Date Range: {summary['date_range']}")
    print(f"Total Missing Values: {summary['total_missing']}")
    
    # Save processed data
    print("\n" + "="*60)
    print("Saving Processed Data")
    print("="*60)
    extractor.save_data(OUTPUT_PATH)
    
    print("\n" + "="*60)
    print("Process Complete!")
    print("="*60)


if __name__ == "__main__":
    main()


# ============================================================
# ADDITIONAL EXAMPLES
# ============================================================

def example_advanced_usage():
    """Examples of more advanced usage patterns."""
    
    extractor = DataExtractor("raw_data_2021_2024.csv")
    extractor.load_data()
    
    # Example 1: Try multiple methods and compare
    print("\n--- Comparing Different Methods ---")
    
    # Get original data
    original = extractor.get_original_data()
    print(f"Original missing values: {original.isna().sum().sum()}")
    
    # Try forward fill
    extractor.handle_missing_data(method='ffill')
    ffill_missing = extractor.get_data().isna().sum().sum()
    print(f"After forward fill: {ffill_missing}")
    
    # Reset and try interpolation
    extractor.reset_to_original()
    extractor.handle_missing_data(method='interpolate', method_type='time')
    interp_missing = extractor.get_data().isna().sum().sum()
    print(f"After interpolation: {interp_missing}")
    
    # Example 2: Polynomial interpolation
    extractor.reset_to_original()
    extractor.handle_missing_data(method='interpolate', method_type='polynomial', order=2)
    
    # Example 3: Drop columns (not rows) with any missing values
    extractor.reset_to_original()
    extractor.handle_missing_data(method='drop', axis=1, how='any')
    
    # Example 4: Drop only rows where ALL values are missing
    extractor.reset_to_original()
    extractor.handle_missing_data(method='drop', axis=0, how='all')


# Uncomment to run advanced examples
# example_advanced_usage()
