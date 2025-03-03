import pandas as pd

def prepare_dates_for_merge(dataframes):
    """
    Convert 'Date' columns to datetime and add 'DateOnly' columns for merging
    
    Args:
        dataframes: List of dataframes that have a 'Date' column
    
    Returns:
        List of dataframes with prepared date columns
    """
    processed_dfs = []
    for df in dataframes:
        if df is not None and not df.empty and 'Date' in df.columns:
            df = df.copy()
            df['Date'] = pd.to_datetime(df['Date'])
            df['DateOnly'] = df['Date'].dt.date
            processed_dfs.append(df)
        else:
            processed_dfs.append(df)
    return processed_dfs