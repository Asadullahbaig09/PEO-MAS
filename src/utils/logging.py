import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(level: str = "INFO", log_file: str = None):
    """
    Setup logging configuration with UTF-8 encoding for Windows
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    
    Returns:
        Configured logger
    """
    # Create logs directory if needed
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Fix for Windows UTF-8 console output
    import io
    if sys.platform == 'win32':
        # Wrap stdout/stderr with UTF-8 encoding
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer,
            encoding='utf-8',
            errors='replace'  # Replace problematic characters
        )
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer,
            encoding='utf-8',
            errors='replace'
        )
    
    # Configure handlers
    handlers = []
    
    # Console handler with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    handlers.append(console_handler)
    
    # File handler with UTF-8 encoding
    if log_file:
        file_handler = logging.FileHandler(
            log_file, 
            encoding='utf-8',  # Explicit UTF-8 for file
            mode='a'
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        handlers=handlers,
        force=True  # Force reconfiguration
    )
    
    logger = logging.getLogger('ethical_mas')
    logger.setLevel(getattr(logging, level.upper()))
    
    return logger


# Alternative: Replace Unicode characters with ASCII
def safe_log_message(message: str) -> str:
    """
    Convert Unicode symbols to ASCII-safe alternatives
    Use this if UTF-8 fix doesn't work
    """
    replacements = {
        '✓': '[OK]',
        '✗': '[ERR]',
        '⚠': '[WARN]',
        '🌟': '[*]',
        '→': '->',
        '←': '<-',
        '↑': '^',
        '↓': 'v'
    }
    
    for unicode_char, ascii_alt in replacements.items():
        message = message.replace(unicode_char, ascii_alt)
    
    return message