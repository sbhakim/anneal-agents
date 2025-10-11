# src/utils/logger.py
"""
Logging configuration for SELFEVOLVE.
Simple setup for console and file logging.
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(config: dict, name: str = "selfevolve") -> logging.Logger:
    """
    Set up logging based on configuration.

    Args:
        config: Logging configuration dictionary
        name: Logger name

    Returns:
        Configured logger instance
    """
    # Get config parameters
    level = config.get('level', 'INFO')
    log_file = config.get('log_file', 'data/logs/selfevolve.log')
    console = config.get('console', True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)

        # Ensure parent directory exists
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            # If it's a file, remove it and create directory
            if log_path.parent.is_file():
                log_path.parent.unlink()
                log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.info(f"Logger initialized. Level: {level}, File: {log_file}")

    return logger


def get_logger(name: str = "selfevolve") -> logging.Logger:
    """
    Get existing logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Testing
if __name__ == "__main__":
    print("Testing logger setup...")

    config = {
        'level': 'INFO',
        'log_file': 'data/logs/test.log',
        'console': True
    }

    logger = setup_logger(config)

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    print("\n✅ Logger test complete. Check data/logs/test.log")