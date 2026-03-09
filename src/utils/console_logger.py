"""
Structured console logging system for Perpetual Ethical Oversight MAS

Provides:
- Colored console output with log levels
- Structured logging format
- Log levels: DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL
- Context-aware message prefixes
- Performance metrics logging
"""

import logging
import sys
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum


class LogLevel(Enum):
    """Log level enumeration"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ColorCodes:
    """ANSI color codes for console output"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    # Foreground colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    
    @staticmethod
    def disable():
        """Disable colors (for non-TTY outputs)"""
        for attr in dir(ColorCodes):
            if not attr.startswith('_') and attr != 'disable':
                setattr(ColorCodes, attr, '')


class ConsoleLogger:
    """
    Structured console logger with colored output and context awareness
    
    Usage:
        logger = ConsoleLogger("system")
        logger.info("Starting system...")
        logger.success("System initialized")
        logger.warning("Low memory warning")
        logger.error("Failed to connect", error=ConnectionError("timeout"))
        logger.critical("System failure", context={"reason": "database down"})
    """
    
    # Map log levels to colors
    LEVEL_COLORS = {
        LogLevel.DEBUG: ColorCodes.CYAN,
        LogLevel.INFO: ColorCodes.BLUE,
        LogLevel.SUCCESS: ColorCodes.GREEN,
        LogLevel.WARNING: ColorCodes.YELLOW,
        LogLevel.ERROR: ColorCodes.RED,
        LogLevel.CRITICAL: ColorCodes.BRIGHT_RED + ColorCodes.BOLD,
    }
    
    # Map log levels to symbols
    LEVEL_SYMBOLS = {
        LogLevel.DEBUG: "🔍",
        LogLevel.INFO: "ℹ️ ",
        LogLevel.SUCCESS: "✅",
        LogLevel.WARNING: "⚠️ ",
        LogLevel.ERROR: "❌",
        LogLevel.CRITICAL: "🚨",
    }
    
    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.INFO,
        use_colors: bool = True,
        show_timestamps: bool = True
    ):
        """
        Initialize console logger
        
        Args:
            name: Logger name/category
            level: Minimum log level to display
            use_colors: Enable colored output
            show_timestamps: Include timestamps in output
        """
        self.name = name
        self.level = level
        self.use_colors = use_colors and sys.stdout.isatty()
        self.show_timestamps = show_timestamps
        self.context_stack = []
        
        # Get underlying Python logger
        self.python_logger = logging.getLogger(f"ethical_mas.{name}")
        self.python_logger.setLevel(logging.DEBUG)
    
    def _format_message(
        self,
        level: LogLevel,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None
    ) -> str:
        """Format message with color and context"""
        timestamp = ""
        if self.show_timestamps:
            timestamp = datetime.now().strftime("%H:%M:%S")
            timestamp = f"{ColorCodes.DIM}[{timestamp}]{ColorCodes.RESET} "
        
        color = self.LEVEL_COLORS.get(level, "")
        reset = ColorCodes.RESET if self.use_colors else ""
        color = color if self.use_colors else ""
        
        symbol = self.LEVEL_SYMBOLS.get(level, "")
        level_str = f"{symbol} {level.value}"
        
        formatted = f"{timestamp}{color}{level_str:<15}{reset} | {self.name:<20} | {message}"
        
        # Add context if provided
        if context:
            context_str = " | ".join(f"{k}={v}" for k, v in context.items())
            formatted += f" | {ColorCodes.DIM}{context_str}{reset}"
        
        # Add error details if provided
        if error:
            error_type = type(error).__name__
            error_msg = str(error)
            error_detail = f" | {ColorCodes.BRIGHT_RED}Error: {error_type}: {error_msg}{reset}"
            formatted += error_detail
        
        return formatted
    
    def _should_log(self, level: LogLevel) -> bool:
        """Check if this log level should be displayed"""
        level_order = [
            LogLevel.DEBUG,
            LogLevel.INFO,
            LogLevel.SUCCESS,
            LogLevel.WARNING,
            LogLevel.ERROR,
            LogLevel.CRITICAL,
        ]
        return level_order.index(level) >= level_order.index(self.level)
    
    def debug(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None
    ):
        """Log debug message"""
        if self._should_log(LogLevel.DEBUG):
            formatted = self._format_message(LogLevel.DEBUG, message, context, error)
            print(formatted, file=sys.stdout)
            self.python_logger.debug(message)
    
    def info(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None
    ):
        """Log info message"""
        if self._should_log(LogLevel.INFO):
            formatted = self._format_message(LogLevel.INFO, message, context, error)
            print(formatted, file=sys.stdout)
            self.python_logger.info(message)
    
    def success(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None
    ):
        """Log success message"""
        if self._should_log(LogLevel.SUCCESS):
            formatted = self._format_message(LogLevel.SUCCESS, message, context, error)
            print(formatted, file=sys.stdout)
            self.python_logger.info(message)
    
    def warning(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None
    ):
        """Log warning message"""
        if self._should_log(LogLevel.WARNING):
            formatted = self._format_message(LogLevel.WARNING, message, context, error)
            print(formatted, file=sys.stderr)
            self.python_logger.warning(message)
    
    def error(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None
    ):
        """Log error message"""
        if self._should_log(LogLevel.ERROR):
            formatted = self._format_message(LogLevel.ERROR, message, context, error)
            print(formatted, file=sys.stderr)
            self.python_logger.error(message, exc_info=error)
    
    def critical(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None
    ):
        """Log critical message"""
        if self._should_log(LogLevel.CRITICAL):
            formatted = self._format_message(LogLevel.CRITICAL, message, context, error)
            print(formatted, file=sys.stderr)
            self.python_logger.critical(message, exc_info=error)
    
    def section(self, title: str):
        """Log section header"""
        width = 80
        line = "=" * width
        color = ColorCodes.BOLD + ColorCodes.BLUE if self.use_colors else ""
        reset = ColorCodes.RESET if self.use_colors else ""
        print(f"\n{color}{line}\n{title.center(width)}\n{line}{reset}\n")
    
    def table(self, data: Dict[str, Any], title: str = ""):
        """Log formatted table"""
        if title:
            self.section(title)
        
        max_key_len = max(len(k) for k in data.keys()) if data else 0
        for key, value in data.items():
            padding = " " * (max_key_len - len(key))
            print(f"  {key}{padding}: {value}")
    
    def metric(self, name: str, value: Any, unit: str = ""):
        """Log performance metric"""
        unit_str = f" {unit}" if unit else ""
        self.info(f"Metric: {name} = {value}{unit_str}")
    
    def push_context(self, **context):
        """Push context to stack for subsequent logs"""
        self.context_stack.append(context)
    
    def pop_context(self):
        """Pop context from stack"""
        if self.context_stack:
            self.context_stack.pop()
    
    def get_context(self) -> Dict[str, Any]:
        """Get merged context from stack"""
        merged = {}
        for ctx in self.context_stack:
            merged.update(ctx)
        return merged


# Global logger instances
_loggers: Dict[str, ConsoleLogger] = {}


def get_logger(name: str, level: LogLevel = LogLevel.INFO) -> ConsoleLogger:
    """
    Get or create a named logger
    
    Args:
        name: Logger name/category
        level: Minimum log level
    
    Returns:
        ConsoleLogger instance
    """
    if name not in _loggers:
        _loggers[name] = ConsoleLogger(name, level=level)
    return _loggers[name]


def set_global_level(level: LogLevel):
    """Set log level for all loggers"""
    for logger in _loggers.values():
        logger.level = level
