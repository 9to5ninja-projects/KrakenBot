"""
System health monitor for KrakenBot.
Monitors system resources, connection status, and bot performance.
"""
import os
import sys
import time
import json
import datetime
import threading
import psutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from exchange import ExchangeManager

class HealthMonitor:
    """Monitors the health of the KrakenBot system."""
    
    def __init__(self):
        """Initialize the health monitor."""
        self.exchange = ExchangeManager(use_api_keys=False)
        self.health_data = {
            'system': {},
            'exchange': {},
            'bot': {}
        }
        self.health_file = config.DATA_DIR / "health.json"
        self.running = False
        self.check_interval = 60  # Check health every 60 seconds
        self.thread = None
        self.last_check_time = None
        self.alert_thresholds = {
            'cpu_percent': 90,
            'memory_percent': 90,
            'disk_percent': 90,
            'api_latency': 5000,  # milliseconds
            'check_interval_drift': 5  # seconds
        }
    
    def start(self):
        """Start the health monitor in a background thread."""
        if self.running:
            logger.warning("Health monitor is already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info("Health monitor started")
    
    def stop(self):
        """Stop the health monitor."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
            self.thread = None
        logger.info("Health monitor stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self.check_health()
                self.save_health_data()
                
                # Sleep until next check
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                time.sleep(10)  # Wait a bit before retrying
    
    def check_health(self):
        """Check the health of the system, exchange, and bot."""
        current_time = datetime.datetime.now()
        self.health_data['timestamp'] = current_time.isoformat()
        
        # Check system health
        self._check_system_health()
        
        # Check exchange health
        self._check_exchange_health()
        
        # Check bot health
        self._check_bot_health()
        
        # Check for alerts
        self._check_alerts()
        
        self.last_check_time = current_time
    
    def _check_system_health(self):
        """Check system resource usage."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used = memory.used / (1024 * 1024 * 1024)  # GB
        memory_total = memory.total / (1024 * 1024 * 1024)  # GB
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        disk_used = disk.used / (1024 * 1024 * 1024)  # GB
        disk_total = disk.total / (1024 * 1024 * 1024)  # GB
        
        # Network stats
        net_io = psutil.net_io_counters()
        
        # Process info
        process = psutil.Process(os.getpid())
        process_cpu = process.cpu_percent(interval=1)
        process_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Update health data
        self.health_data['system'] = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'memory_used_gb': round(memory_used, 2),
            'memory_total_gb': round(memory_total, 2),
            'disk_percent': disk_percent,
            'disk_used_gb': round(disk_used, 2),
            'disk_total_gb': round(disk_total, 2),
            'net_bytes_sent': net_io.bytes_sent,
            'net_bytes_recv': net_io.bytes_recv,
            'process_cpu_percent': process_cpu,
            'process_memory_mb': round(process_memory, 2),
            'uptime_seconds': time.time() - psutil.boot_time()
        }
    
    def _check_exchange_health(self):
        """Check exchange connection and API status."""
        start_time = time.time()
        
        try:
            # Test API connection with a simple request
            ticker = self.exchange.fetch_ticker('BTC/CAD')
            api_latency = (time.time() - start_time) * 1000  # milliseconds
            
            # Update health data
            self.health_data['exchange'] = {
                'status': 'online',
                'api_latency_ms': round(api_latency, 2),
                'last_price': ticker.get('last', 0),
                'timestamp': ticker.get('timestamp', 0)
            }
        except Exception as e:
            # Update health data with error
            self.health_data['exchange'] = {
                'status': 'error',
                'error': str(e),
                'api_latency_ms': (time.time() - start_time) * 1000
            }
    
    def _check_bot_health(self):
        """Check bot performance and status."""
        # Get data directory stats
        data_dir = config.DATA_DIR
        log_dir = config.LOG_DIR
        
        data_files = list(data_dir.glob('*'))
        log_files = list(log_dir.glob('*'))
        
        # Get latest opportunity data
        opportunities_file = data_dir / "opportunities.csv"
        last_opportunity_time = None
        opportunity_count = 0
        
        if opportunities_file.exists():
            try:
                # Get file modification time
                last_opportunity_time = datetime.datetime.fromtimestamp(
                    opportunities_file.stat().st_mtime
                ).isoformat()
                
                # Count lines (opportunities)
                with open(opportunities_file, 'r') as f:
                    # Subtract 1 for header
                    opportunity_count = sum(1 for _ in f) - 1
            except Exception as e:
                logger.error(f"Error reading opportunities file: {e}")
        
        # Check interval drift if we have a previous check time
        interval_drift = None
        if self.last_check_time:
            actual_interval = (datetime.datetime.now() - self.last_check_time).total_seconds()
            interval_drift = actual_interval - self.check_interval
        
        # Update health data
        self.health_data['bot'] = {
            'data_file_count': len(data_files),
            'log_file_count': len(log_files),
            'last_opportunity_time': last_opportunity_time,
            'opportunity_count': opportunity_count,
            'check_interval': self.check_interval,
            'interval_drift': interval_drift,
            'trading_mode': config.TRADING_MODE
        }
    
    def _check_alerts(self):
        """Check for alert conditions."""
        alerts = []
        
        # System alerts
        if self.health_data['system'].get('cpu_percent', 0) > self.alert_thresholds['cpu_percent']:
            alerts.append(f"High CPU usage: {self.health_data['system']['cpu_percent']}%")
        
        if self.health_data['system'].get('memory_percent', 0) > self.alert_thresholds['memory_percent']:
            alerts.append(f"High memory usage: {self.health_data['system']['memory_percent']}%")
        
        if self.health_data['system'].get('disk_percent', 0) > self.alert_thresholds['disk_percent']:
            alerts.append(f"High disk usage: {self.health_data['system']['disk_percent']}%")
        
        # Exchange alerts
        if self.health_data['exchange'].get('status') == 'error':
            alerts.append(f"Exchange connection error: {self.health_data['exchange'].get('error')}")
        
        if self.health_data['exchange'].get('api_latency_ms', 0) > self.alert_thresholds['api_latency']:
            alerts.append(f"High API latency: {self.health_data['exchange']['api_latency_ms']} ms")
        
        # Bot alerts
        if self.health_data['bot'].get('interval_drift') and \
           abs(self.health_data['bot']['interval_drift']) > self.alert_thresholds['check_interval_drift']:
            alerts.append(f"Check interval drift: {self.health_data['bot']['interval_drift']} seconds")
        
        # Add alerts to health data
        self.health_data['alerts'] = alerts
        
        # Log alerts
        for alert in alerts:
            logger.warning(f"Health alert: {alert}")
    
    def save_health_data(self):
        """Save health data to a JSON file."""
        try:
            with open(self.health_file, 'w') as f:
                json.dump(self.health_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving health data: {e}")
    
    def get_health_data(self):
        """Get the current health data."""
        return self.health_data
    
    def get_health_summary(self):
        """Get a summary of the health status."""
        if not self.health_data:
            return "Health data not available"
        
        system = self.health_data.get('system', {})
        exchange = self.health_data.get('exchange', {})
        bot = self.health_data.get('bot', {})
        alerts = self.health_data.get('alerts', [])
        
        summary = []
        summary.append(f"Time: {self.health_data.get('timestamp')}")
        summary.append(f"System: CPU {system.get('cpu_percent', 0)}%, "
                      f"Memory {system.get('memory_percent', 0)}%, "
                      f"Disk {system.get('disk_percent', 0)}%")
        
        summary.append(f"Exchange: {exchange.get('status', 'unknown')}, "
                      f"Latency {exchange.get('api_latency_ms', 0)} ms")
        
        summary.append(f"Bot: Mode {bot.get('trading_mode', 'unknown')}, "
                      f"Opportunities {bot.get('opportunity_count', 0)}")
        
        if alerts:
            summary.append(f"Alerts: {len(alerts)}")
            for alert in alerts:
                summary.append(f"  - {alert}")
        else:
            summary.append("Alerts: None")
        
        return "\n".join(summary)


# Singleton instance
_monitor_instance = None

def get_monitor():
    """Get the singleton health monitor instance."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = HealthMonitor()
    return _monitor_instance

if __name__ == "__main__":
    # Create data and log directories if they don't exist
    config.DATA_DIR.mkdir(exist_ok=True)
    config.LOG_DIR.mkdir(exist_ok=True)
    
    # Configure logger
    logger.remove()
    logger.add(
        config.LOG_DIR / "health_monitor.log",
        rotation="1 day",
        retention="7 days",
        level="INFO"
    )
    logger.add(lambda msg: print(msg), level="INFO")
    
    # Run a single health check
    monitor = get_monitor()
    monitor.check_health()
    monitor.save_health_data()
    
    print("\nHealth Summary:")
    print("=" * 80)
    print(monitor.get_health_summary())
    print("=" * 80)
    print(f"\nDetailed health data saved to {monitor.health_file}")
    
    # Ask if user wants to start continuous monitoring
    if input("\nStart continuous health monitoring? (y/n): ").lower() == 'y':
        try:
            monitor.start()
            print("Health monitor started. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            monitor.stop()
            print("\nHealth monitor stopped.")