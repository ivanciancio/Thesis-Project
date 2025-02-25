import streamlit as st
import logging
from functools import wraps

def handle_api_error(func):
    """Decorator to handle API errors"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"Error in {func.__name__}: {str(e)}"
            logging.error(error_msg)
            st.error(error_msg)
            return None
    return wrapper