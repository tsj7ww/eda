import os
import sys
import time
import json
import hashlib
import uuid
import pickle
import logging
import traceback
from typing import Dict, List, Tuple, Union, Optional, Any, Callable, TypeVar, Generic
from functools import wraps
from contextlib import contextmanager
import pandas as pd
import numpy as np
from datetime import datetime

# Type variables for generics
T = TypeVar('T')


# Logging configuration
def setup_logging(
    log_level: str = "INFO",
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Configure logging for the fraud detection system.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format for log messages
        log_file: Optional file path to write logs to
        console_output: Whether to output logs to console
        
    Returns:
        Root logger instance
    """
    # Map string level to logging constant
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    numeric_level = level_map.get(log_level.upper(), logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters
    formatter = logging.Formatter(log_format)
    
    # Add file handler if specified
    if log_file:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    return root_logger


# Performance monitoring
@contextmanager
def timer(name: str = None, logger: logging.Logger = None):
    """
    Context manager for timing code execution.
    
    Args:
        name: Name of the operation being timed
        logger: Logger to use (if None, prints to stdout)
        
    Yields:
        None
    """
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    
    message = f"Time elapsed for {name or 'operation'}: {elapsed_time:.2f} seconds"
    
    if logger:
        logger.info(message)
    else:
        print(message)


def timeit(func: Callable) -> Callable:
    """
    Decorator for timing function execution.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        
        # Get function name
        fn_name = func.__name__
        
        # Log the timing
        logger = logging.getLogger(func.__module__)
        logger.info(f"Function {fn_name} executed in {elapsed_time:.2f} seconds")
        
        return result
    
    return wrapper


class MemoryTracker:
    """Utility for tracking memory usage of objects."""
    
    @staticmethod
    def get_size(obj: Any, seen: Optional[set] = None) -> int:
        """
        Recursively calculate size of objects in bytes.
        
        Args:
            obj: Object to calculate size for
            seen: Set of object ids already processed
            
        Returns:
            Size in bytes
        """
        import sys
        
        # Handle recursion by tracking objects already seen
        if seen is None:
            seen = set()
            
        # Get object id and check if already processed
        obj_id = id(obj)
        if obj_id in seen:
            return 0
            
        # Mark as seen
        seen.add(obj_id)
        
        # Get size of the object itself
        size = sys.getsizeof(obj)
        
        # Handle different types of objects for recursive size calculation
        if isinstance(obj, dict):
            size += sum(MemoryTracker.get_size(k, seen) + MemoryTracker.get_size(v, seen) for k, v in obj.items())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            size += sum(MemoryTracker.get_size(i, seen) for i in obj)
        elif isinstance(obj, np.ndarray):
            size = obj.nbytes
        elif isinstance(obj, pd.DataFrame):
            size = obj.memory_usage(deep=True).sum()
        
        return size
    
    @staticmethod
    def format_bytes(bytes: int) -> str:
        """
        Format bytes as human-readable string.
        
        Args:
            bytes: Number of bytes
            
        Returns:
            Formatted string (e.g., "1.23 MB")
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes < 1024 or unit == 'TB':
                return f"{bytes:.2f} {unit}"
            bytes /= 1024
    
    @staticmethod
    def log_memory_usage(obj: Any, name: str, logger: Optional[logging.Logger] = None):
        """
        Log memory usage of an object.
        
        Args:
            obj: Object to check memory usage of
            name: Name to identify the object in the log
            logger: Logger to use (if None, prints to stdout)
        """
        size_bytes = MemoryTracker.get_size(obj)
        formatted_size = MemoryTracker.format_bytes(size_bytes)
        
        message = f"Memory usage of {name}: {formatted_size}"
        
        if logger:
            logger.info(message)
        else:
            print(message)


# Error handling
class ErrorHandler:
    """Standardized error handling utilities."""
    
    @staticmethod
    def log_exception(
        logger: logging.Logger,
        exception: Exception,
        message: str = "An error occurred",
        include_traceback: bool = True,
        include_context: Dict[str, Any] = None
    ) -> None:
        """
        Log an exception with consistent formatting.
        
        Args:
            logger: Logger to use
            exception: Exception to log
            message: Message to log with the exception
            include_traceback: Whether to include traceback in the log
            include_context: Optional context information to include
        """
        # Basic error message
        error_message = f"{message}: {str(exception)}"
        
        # Add context information if provided
        if include_context:
            context_str = ", ".join(f"{k}={v}" for k, v in include_context.items())
            error_message += f" [Context: {context_str}]"
        
        # Log the error
        logger.error(error_message)
        
        # Add traceback if requested
        if include_traceback:
            logger.error("Traceback:", exc_info=True)
    
    @staticmethod
    def safe_execute(
        func: Callable,
        args: List = None,
        kwargs: Dict = None,
        logger: logging.Logger = None,
        default_return: Any = None,
        error_message: str = "Function execution failed",
        raise_error: bool = False
    ) -> Any:
        """
        Execute a function with error handling.
        
        Args:
            func: Function to execute
            args: Positional arguments to pass to the function
            kwargs: Keyword arguments to pass to the function
            logger: Logger to use for errors
            default_return: Value to return if function fails
            error_message: Message to log if function fails
            raise_error: Whether to re-raise the exception
            
        Returns:
            Function result or default_return if failed
        """
        args = args or []
        kwargs = kwargs or {}
        
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if logger:
                ErrorHandler.log_exception(
                    logger,
                    e,
                    message=error_message,
                    include_context={
                        "function": func.__name__,
                        "args": str(args),
                        "kwargs": str(kwargs)
                    }
                )
            
            if raise_error:
                raise
                
            return default_return
    
    @staticmethod
    def retry(
        max_attempts: int = 3,
        delay: float = 1.0,
        backoff: float = 2.0,
        exceptions: Tuple[Exception] = (Exception,),
        logger: logging.Logger = None
    ) -> Callable:
        """
        Create a retry decorator for a function.
        
        Args:
            max_attempts: Maximum number of attempts
            delay: Initial delay between attempts in seconds
            backoff: Backoff factor for increasing delay
            exceptions: Tuple of exceptions to catch and retry
            logger: Logger to use
            
        Returns:
            Retry decorator
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                current_delay = delay
                last_exception = None
                
                for attempt in range(1, max_attempts + 1):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        
                        if logger:
                            logger.warning(
                                f"Attempt {attempt}/{max_attempts} for {func.__name__} failed: {str(e)}"
                            )
                        
                        if attempt < max_attempts:
                            if logger:
                                logger.info(f"Retrying in {current_delay:.2f} seconds...")
                            
                            time.sleep(current_delay)
                            current_delay *= backoff
                
                # If we get here, all attempts failed
                if logger:
                    ErrorHandler.log_exception(
                        logger,
                        last_exception,
                        message=f"All {max_attempts} attempts failed for {func.__name__}",
                        include_context={
                            "args": str(args),
                            "kwargs": str(kwargs)
                        }
                    )
                
                raise last_exception
            
            return wrapper
        
        return decorator