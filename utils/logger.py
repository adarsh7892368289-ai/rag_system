"""
Logging Utilities - Clean Production Version

Provides structured logging for the RAG system.
"""

import logging
import time
from pathlib import Path
from typing import Optional
from datetime import datetime


# Create logs directory
LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get configured logger
    
    Args:
        name: Logger name (usually module name)
        level: Logging level
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = LOGS_DIR / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger


class PerformanceLogger:
    """
    Context manager for performance logging
    
    Usage:
        with PerformanceLogger("Operation name"):
            # Your code here
            pass
    """
    
    def __init__(self, operation_name: str, logger_name: str = "performance"):
        """
        Initialize performance logger
        
        Args:
            operation_name: Name of operation being timed
            logger_name: Logger name to use
        """
        self.operation_name = operation_name
        self.logger = get_logger(logger_name)
        self.start_time = None
    
    def __enter__(self):
        """Start timing"""
        self.start_time = time.time()
        self.logger.debug(f"⏱️  Started: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and log duration"""
        if self.start_time:
            elapsed = time.time() - self.start_time
            
            if elapsed < 1:
                time_str = f"{elapsed*1000:.0f}ms"
            elif elapsed < 60:
                time_str = f"{elapsed:.2f}s"
            else:
                minutes = int(elapsed // 60)
                seconds = elapsed % 60
                time_str = f"{minutes}m {seconds:.1f}s"
            
            self.logger.info(f"✅ Completed: {self.operation_name} ({time_str})")
        
        return False


class ProgressLogger:
    """
    Progress logging for long operations
    
    Usage:
        progress = ProgressLogger("Processing files", total=100)
        for i in range(100):
            progress.update(i+1, f"Processing file {i+1}")
        progress.complete("All files processed")
    """
    
    def __init__(self, operation_name: str, total: int, logger_name: str = "progress"):
        """
        Initialize progress logger
        
        Args:
            operation_name: Name of operation
            total: Total number of items
            logger_name: Logger name to use
        """
        self.operation_name = operation_name
        self.total = total
        self.logger = get_logger(logger_name)
        self.start_time = time.time()
        
        self.logger.info(f"🔄 Starting: {self.operation_name} (total={total})")
    
    def update(self, current: int, message: Optional[str] = None):
        """
        Update progress
        
        Args:
            current: Current item number
            message: Optional status message
        """
        percentage = (current / self.total * 100) if self.total > 0 else 0
        elapsed = time.time() - self.start_time
        
        # Estimate time remaining
        if current > 0:
            eta_seconds = (elapsed / current) * (self.total - current)
            if eta_seconds < 60:
                eta_str = f"ETA: {eta_seconds:.0f}s"
            else:
                eta_minutes = int(eta_seconds // 60)
                eta_str = f"ETA: {eta_minutes}m"
        else:
            eta_str = "ETA: calculating..."
        
        status = f"[{current}/{self.total}] {percentage:.1f}% - {eta_str}"
        if message:
            status += f" - {message}"
        
        self.logger.info(status)
    
    def complete(self, message: Optional[str] = None):
        """
        Mark operation as complete
        
        Args:
            message: Optional completion message
        """
        elapsed = time.time() - self.start_time
        
        if elapsed < 60:
            time_str = f"{elapsed:.2f}s"
        else:
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            time_str = f"{minutes}m {seconds:.1f}s"
        
        status = f"✅ {self.operation_name} complete ({time_str})"
        if message:
            status += f" - {message}"
        
        self.logger.info(status)


# Convenience function for simple timing
def log_execution_time(func):
    """
    Decorator to log function execution time
    
    Usage:
        @log_execution_time
        def my_function():
            pass
    """
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"⏱️  {func.__name__} completed in {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ {func.__name__} failed after {elapsed:.2f}s: {e}")
            raise
    
    return wrapper


if __name__ == "__main__":
    # Demo
    logger = get_logger("demo")
    
    logger.info("This is an info message")
    logger.warning("This is a warning")
    logger.error("This is an error")
    
    # Performance logging
    with PerformanceLogger("Test operation"):
        time.sleep(1)
    
    # Progress logging
    progress = ProgressLogger("Test progress", total=10)
    for i in range(10):
        time.sleep(0.1)
        progress.update(i + 1, f"Item {i + 1}")
    progress.complete("All items processed")
    
    # Decorator
    @log_execution_time
    def test_function():
        time.sleep(0.5)
        return "Done"
    
    test_function()