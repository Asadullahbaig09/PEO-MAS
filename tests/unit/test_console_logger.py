"""
Tests for console logging system
"""

import pytest
from io import StringIO
import sys
from datetime import datetime

from src.utils.console_logger import (
    ConsoleLogger,
    LogLevel,
    get_logger,
    set_global_level,
    ColorCodes
)


class TestConsoleLogger:
    """Test console logger functionality"""
    
    def test_logger_creation(self):
        """Test logger initialization"""
        logger = ConsoleLogger("test")
        assert logger.name == "test"
        assert logger.level == LogLevel.INFO
    
    def test_logger_debug_level(self):
        """Test debug level messages"""
        logger = ConsoleLogger("test", level=LogLevel.DEBUG)
        assert logger._should_log(LogLevel.DEBUG)
    
    def test_logger_info_level(self):
        """Test info level messages"""
        logger = ConsoleLogger("test", level=LogLevel.INFO)
        assert logger._should_log(LogLevel.INFO)
        assert not logger._should_log(LogLevel.DEBUG)
    
    def test_logger_warning_level(self):
        """Test warning level threshold"""
        logger = ConsoleLogger("test", level=LogLevel.WARNING)
        assert logger._should_log(LogLevel.WARNING)
        assert logger._should_log(LogLevel.ERROR)
        assert not logger._should_log(LogLevel.INFO)
    
    def test_logger_error_level(self):
        """Test error level threshold"""
        logger = ConsoleLogger("test", level=LogLevel.ERROR)
        assert logger._should_log(LogLevel.ERROR)
        assert logger._should_log(LogLevel.CRITICAL)
        assert not logger._should_log(LogLevel.WARNING)
    
    def test_logger_critical_level(self):
        """Test critical level threshold"""
        logger = ConsoleLogger("test", level=LogLevel.CRITICAL)
        assert logger._should_log(LogLevel.CRITICAL)
        assert not logger._should_log(LogLevel.ERROR)
    
    def test_message_formatting(self, capsys):
        """Test message formatting"""
        logger = ConsoleLogger("test", level=LogLevel.INFO, show_timestamps=False)
        logger.info("Test message")
        captured = capsys.readouterr()
        assert "Test message" in captured.out
        assert "test" in captured.out
    
    def test_debug_logging(self, capsys):
        """Test debug message logging"""
        logger = ConsoleLogger("debug_test", level=LogLevel.DEBUG)
        logger.debug("Debug message")
        captured = capsys.readouterr()
        assert "Debug message" in captured.out
    
    def test_info_logging(self, capsys):
        """Test info message logging"""
        logger = ConsoleLogger("info_test", level=LogLevel.INFO)
        logger.info("Info message")
        captured = capsys.readouterr()
        assert "Info message" in captured.out
    
    def test_success_logging(self, capsys):
        """Test success message logging"""
        logger = ConsoleLogger("success_test", level=LogLevel.SUCCESS)
        logger.success("Success message")
        captured = capsys.readouterr()
        assert "Success message" in captured.out
    
    def test_warning_logging(self, capsys):
        """Test warning message logging"""
        logger = ConsoleLogger("warning_test", level=LogLevel.WARNING)
        logger.warning("Warning message")
        captured = capsys.readouterr()
        assert "Warning message" in captured.err
    
    def test_error_logging(self, capsys):
        """Test error message logging"""
        logger = ConsoleLogger("error_test", level=LogLevel.ERROR)
        logger.error("Error message")
        captured = capsys.readouterr()
        assert "Error message" in captured.err
    
    def test_critical_logging(self, capsys):
        """Test critical message logging"""
        logger = ConsoleLogger("critical_test", level=LogLevel.CRITICAL)
        logger.critical("Critical message")
        captured = capsys.readouterr()
        assert "Critical message" in captured.err
    
    def test_logging_with_context(self, capsys):
        """Test logging with context data"""
        logger = ConsoleLogger("context_test", level=LogLevel.INFO)
        logger.info("Message", context={"key": "value", "num": 42})
        captured = capsys.readouterr()
        assert "key=value" in captured.out or "num=42" in captured.out
    
    def test_logging_with_error(self, capsys):
        """Test logging with exception"""
        logger = ConsoleLogger("error_context_test", level=LogLevel.ERROR)
        try:
            raise ValueError("Test error")
        except ValueError as e:
            logger.error("An error occurred", error=e)
        captured = capsys.readouterr()
        assert "ValueError" in captured.err or "Test error" in captured.err
    
    def test_color_codes_present(self):
        """Test color codes are defined"""
        assert ColorCodes.RED != ""
        assert ColorCodes.GREEN != ""
        assert ColorCodes.YELLOW != ""
        assert ColorCodes.RESET != ""
    
    def test_color_codes_disable(self):
        """Test disabling color codes"""
        ColorCodes.disable()
        assert ColorCodes.RED == ""
        assert ColorCodes.GREEN == ""
    
    def test_context_stack(self):
        """Test context stack operations"""
        logger = ConsoleLogger("stack_test", level=LogLevel.INFO)
        
        logger.push_context(user="alice")
        ctx = logger.get_context()
        assert ctx == {"user": "alice"}
        
        logger.push_context(action="read")
        ctx = logger.get_context()
        assert ctx == {"user": "alice", "action": "read"}
        
        logger.pop_context()
        ctx = logger.get_context()
        assert ctx == {"user": "alice"}
    
    def test_global_logger_retrieval(self):
        """Test getting named loggers"""
        logger1 = get_logger("app")
        logger2 = get_logger("app")
        
        assert logger1 is logger2
    
    def test_different_loggers(self):
        """Test multiple different loggers"""
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")
        
        assert logger1 is not logger2
        assert logger1.name == "module1"
        assert logger2.name == "module2"
    
    def test_global_level_setting(self):
        """Test setting global log level"""
        logger1 = get_logger("test1", level=LogLevel.INFO)
        logger2 = get_logger("test2", level=LogLevel.INFO)
        
        set_global_level(LogLevel.WARNING)
        
        assert logger1.level == LogLevel.WARNING
        assert logger2.level == LogLevel.WARNING
    
    def test_message_without_timestamp(self, capsys):
        """Test message without timestamp"""
        logger = ConsoleLogger("no_time", level=LogLevel.INFO, show_timestamps=False)
        logger.info("No timestamp")
        captured = capsys.readouterr()
        # Should not have time format
        assert "No timestamp" in captured.out
    
    def test_message_with_timestamp(self, capsys):
        """Test message with timestamp"""
        logger = ConsoleLogger("with_time", level=LogLevel.INFO, show_timestamps=True)
        logger.info("With timestamp")
        captured = capsys.readouterr()
        assert "With timestamp" in captured.out


class TestLogLevelEnum:
    """Test LogLevel enumeration"""
    
    def test_log_levels_defined(self):
        """Test all log levels are defined"""
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.SUCCESS.value == "SUCCESS"
        assert LogLevel.WARNING.value == "WARNING"
        assert LogLevel.ERROR.value == "ERROR"
        assert LogLevel.CRITICAL.value == "CRITICAL"
    
    def test_log_level_comparison(self):
        """Test log level comparison"""
        assert LogLevel.DEBUG != LogLevel.INFO
        assert LogLevel.ERROR != LogLevel.WARNING


class TestLoggingIntegration:
    """Integration tests for logging system"""
    
    def test_multi_logger_scenario(self, capsys):
        """Test multiple loggers in operation"""
        system_logger = get_logger("system", level=LogLevel.INFO)
        api_logger = get_logger("api", level=LogLevel.DEBUG)
        
        system_logger.info("System started")
        api_logger.debug("API initialized")
        
        captured = capsys.readouterr()
        assert "System started" in captured.out
        assert "API initialized" in captured.out
    
    def test_error_tracking_workflow(self, capsys):
        """Test typical error tracking workflow"""
        logger = get_logger("workflow", level=LogLevel.ERROR)
        
        logger.info("Processing request")  # Won't show due to level
        
        try:
            raise RuntimeError("Processing failed")
        except RuntimeError as e:
            logger.error("Request failed", error=e)
        
        captured = capsys.readouterr()
        assert "RuntimeError" in captured.err or "Processing failed" in captured.err
    
    def test_success_milestone_logging(self, capsys):
        """Test logging success milestones"""
        logger = get_logger("milestones", level=LogLevel.SUCCESS)
        
        logger.success("Phase 1 complete")
        logger.success("Phase 2 complete")
        
        captured = capsys.readouterr()
        assert "Phase 1 complete" in captured.out
        assert "Phase 2 complete" in captured.out
