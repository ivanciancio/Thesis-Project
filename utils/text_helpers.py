def find_sentiment_column(dataframe):
    """Find sentiment score column in a dataframe"""
    for col in dataframe.columns:
        if 'sentiment' in col.lower() and 'score' in col.lower():
            return col
    return None