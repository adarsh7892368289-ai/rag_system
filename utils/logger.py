"""
Structured Logging System

Provides consistent, colorful logging across the entire RAG system.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional
from datetime import datetime

# ANSI color codes for terminal
COLORS = {
    'DEBUG': '\033[36m',     # Cyan
    'INFO': '\033[32m',      # Green
    'WARNING': '\033[33m',   # Yellow
    'ERROR': '\033[31m',     # Red
    'CRITICAL': '\033[35m',  # Magenta
    'RESET': '\033[0m'       # Reset
}

# Emojis for visual clarity
EMOJIS = {
    'DEBUG': '🔍',
    'INFO': '✅',
    'WARNING': '⚠️',
    'ERROR': '❌',
    'CRITICAL': '🚨'
}


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors and emojis"""
    
    def format(self, record):
        # Add color
        levelname = record.levelname
        if levelname in COLORS:
            record.levelname = f"{COLORS[levelname]}{EMOJIS.get(levelname, '')} {levelname}{COLORS['RESET']}"
        
        # Format the message
        return super().format(record)


class RAGLogger:
    """
    Centralized logger for the RAG system
    
    Features:
    - Colored console output
    - File logging with rotation
    - Module-specific loggers
    - Performance tracking
    """
    
    _loggers = {}
    _initialized = False
    
    @classmethod
    def setup(cls, 
              log_level: str = "INFO",
              log_to_file: bool = True,
              log_file: str = "logs/rag_system.log",
              max_bytes: int = 10 * 1024 * 1024,
              backup_count: int = 5):
        """
        Setup the logging system (call once at startup)
        
        Args:
            log_level: DEBUG, INFO, WARNING, ERROR, CRITICAL
            log_to_file: Enable file logging
            log_file: Path to log file
            max_bytes: Max log file size before rotation
            backup_count: Number of backup files to keep
        """
        if cls._initialized:
            return
        
        # Create logs directory
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_formatter = ColoredFormatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # File handler (no colors, with rotation)
        if log_to_file:
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)  # Log everything to file
            file_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
        
        cls._initialized = True
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get a logger for a specific module
        
        Args:
            name: Module name (e.g., 'chunking', 'extraction', 'search')
        
        Returns:
            Configured logger instance
        """
        if not cls._initialized:
            cls.setup()
        
        if name not in cls._loggers:
            cls._loggers[name] = logging.getLogger(name)
        
        return cls._loggers[name]


class ProgressLogger:
    """
    Logger for tracking progress of long-running operations
    
    Usage:
        progress = ProgressLogger("Processing documents", total=100)
        for i in range(100):
            progress.update(i + 1)
        progress.complete()
    """
    
    def __init__(self, task_name: str, total: int, logger: Optional[logging.Logger] = None):
        self.task_name = task_name
        self.total = total
        self.current = 0
        self.logger = logger or RAGLogger.get_logger("progress")
        self.start_time = datetime.now()
        
        self.logger.info(f"🚀 Starting: {task_name} (total: {total})")
    
    def update(self, current: int, message: str = ""):
        """Update progress"""
        self.current = current
        percentage = (current / self.total * 100) if self.total > 0 else 0
        
        # Calculate ETA
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if current > 0:
            eta_seconds = (elapsed / current) * (self.total - current)
            eta = f"ETA: {int(eta_seconds)}s"
        else:
            eta = "ETA: calculating..."
        
        msg = f"   [{current}/{self.total}] {percentage:.1f}% - {eta}"
        if message:
            msg += f" - {message}"
        
        self.logger.info(msg)
    
    def complete(self, message: str = ""):
        """Mark as complete"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        msg = f"✅ Completed: {self.task_name} in {elapsed:.2f}s"
        if message:
            msg += f" - {message}"
        self.logger.info(msg)


class PerformanceLogger:
    """
    Context manager for logging performance metrics
    
    Usage:
        with PerformanceLogger("Embedding generation"):
            # Your code here
            pass
    """
    
    def __init__(self, operation_name: str, logger: Optional[logging.Logger] = None):
        self.operation_name = operation_name
        self.logger = logger or RAGLogger.get_logger("performance")
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.debug(f"⏱️  Started: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.debug(f"⏱️  Completed: {self.operation_name} in {elapsed:.3f}s")
        else:
            self.logger.error(f"❌ Failed: {self.operation_name} after {elapsed:.3f}s")
        
        return False  # Don't suppress exceptions


# Convenience functions
def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    return RAGLogger.get_logger(name)


def log_section(logger: logging.Logger, title: str, width: int = 70):
    """Log a section header"""
    logger.info("=" * width)
    logger.info(title.center(width))
    logger.info("=" * width)


def log_config(logger: logging.Logger, config_dict: dict, title: str = "Configuration"):
    """Log configuration in a readable format"""
    logger.info(f"\n{'='*50}")
    logger.info(f"{title}")
    logger.info(f"{'='*50}")
    for key, value in config_dict.items():
        logger.info(f"  {key}: {value}")
    logger.info(f"{'='*50}\n")


# Initialize on import with default settings
RAGLogger.setup()


if __name__ == "__main__":
    # Demo
    logger = get_logger("demo")
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning")
    logger.error("This is an error")
    
    # Progress demo
    progress = ProgressLogger("Test Task", total=10)
    import time
    for i in range(10):
        time.sleep(0.1)
        progress.update(i + 1, f"Processing item {i+1}")
    progress.complete("All items processed")
    
    # Performance demo
    with PerformanceLogger("Heavy computation"):
        time.sleep(0.5)