"""
KrakenBot AI Module
Advanced AI capabilities for trading intelligence
"""

__version__ = "1.0.0"
__author__ = "KrakenBot AI Team"

from .nlp.log_analyzer import TradingLogAnalyzer
from .nlp.report_generator import AIReportGenerator

__all__ = [
    'TradingLogAnalyzer',
    'AIReportGenerator'
]