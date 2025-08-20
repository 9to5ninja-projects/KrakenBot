"""
Notification module for KrakenBot.
Handles sending notifications for profitable arbitrage opportunities.
"""
import os
import sys
import json
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from arbitrage import ArbitrageOpportunity

class NotificationManager:
    """Manages notifications for KrakenBot."""
    
    def __init__(self):
        """Initialize the notification manager."""
        # Load notification settings from environment variables
        self.enabled = os.getenv('ENABLE_NOTIFICATIONS', 'true').lower() == 'true'
        self.method = os.getenv('NOTIFICATION_METHOD', 'Console Only')
        
        # Email settings
        self.smtp_server = os.getenv('SMTP_SERVER', '')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.smtp_username = os.getenv('SMTP_USERNAME', '')
        self.smtp_password = os.getenv('SMTP_PASSWORD', '')
        self.recipient_email = os.getenv('RECIPIENT_EMAIL', '')
        
        # Telegram settings
        self.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID', '')
        
        # Discord settings
        self.discord_webhook_url = os.getenv('DISCORD_WEBHOOK_URL', '')
        
        # Notification history
        self.notification_count = 0
        self.last_notification_time = None
        self.notification_history_file = config.DATA_DIR / "notification_history.json"
        self.notification_history = self._load_notification_history()
        
        # Notification throttling
        self.min_notification_interval = int(os.getenv('MIN_NOTIFICATION_INTERVAL', '300'))  # 5 minutes
        
        logger.info(f"Notification manager initialized with method: {self.method}")
        if not self.enabled:
            logger.info("Notifications are disabled")
    
    def _load_notification_history(self):
        """Load notification history from file."""
        if self.notification_history_file.exists():
            try:
                with open(self.notification_history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading notification history: {e}")
        
        return {
            'count': 0,
            'history': []
        }
    
    def _save_notification_history(self):
        """Save notification history to file."""
        try:
            with open(self.notification_history_file, 'w') as f:
                json.dump(self.notification_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving notification history: {e}")
    
    def _format_opportunity_message(self, opportunity: ArbitrageOpportunity, include_html=False):
        """Format an arbitrage opportunity as a notification message."""
        timestamp = opportunity.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        
        if include_html:
            message = f"""
            <h2>🚀 Profitable Arbitrage Opportunity</h2>
            <p><strong>Time:</strong> {timestamp}</p>
            <p><strong>Path:</strong> {' → '.join(opportunity.path)}</p>
            <p><strong>Start Amount:</strong> ${opportunity.start_amount:.2f}</p>
            <p><strong>End Amount:</strong> ${opportunity.end_amount:.2f}</p>
            <p><strong>Profit:</strong> ${opportunity.profit:.2f} ({opportunity.profit_percentage:.4f}%)</p>
            <h3>Prices:</h3>
            <ul>
            """
            
            for symbol, price in opportunity.prices.items():
                message += f"<li><strong>{symbol}:</strong> {price}</li>"
            
            message += """
            </ul>
            <p>This is an automated notification from KrakenBot.</p>
            """
        else:
            message = f"""
🚀 Profitable Arbitrage Opportunity
Time: {timestamp}
Path: {' → '.join(opportunity.path)}
Start Amount: ${opportunity.start_amount:.2f}
End Amount: ${opportunity.end_amount:.2f}
Profit: ${opportunity.profit:.2f} ({opportunity.profit_percentage:.4f}%)

Prices:
"""
            
            for symbol, price in opportunity.prices.items():
                message += f"- {symbol}: {price}\n"
            
            message += "\nThis is an automated notification from KrakenBot."
        
        return message
    
    def send_notification(self, opportunity: ArbitrageOpportunity):
        """Send a notification for a profitable arbitrage opportunity."""
        if not self.enabled:
            logger.debug("Notifications are disabled")
            return False
        
        if not opportunity.is_profitable:
            logger.debug("Not sending notification for non-profitable opportunity")
            return False
        
        # Format the message
        plain_message = self._format_opportunity_message(opportunity, include_html=False)
        html_message = self._format_opportunity_message(opportunity, include_html=True)
        
        # Send notification based on method
        success = False
        
        if self.method == 'Console Only':
            logger.info(plain_message)
            success = True
        
        elif self.method == 'Email':
            success = self._send_email_notification(
                subject=f"KrakenBot: Profitable Opportunity (${opportunity.profit:.2f})",
                plain_message=plain_message,
                html_message=html_message
            )
        
        elif self.method == 'Telegram':
            success = self._send_telegram_notification(plain_message)
        
        elif self.method == 'Discord':
            success = self._send_discord_notification(
                title=f"KrakenBot: Profitable Opportunity (${opportunity.profit:.2f})",
                message=plain_message
            )
        
        # Update notification history
        if success:
            self.notification_count += 1
            self.last_notification_time = opportunity.timestamp
            
            # Add to history
            self.notification_history['count'] += 1
            self.notification_history['history'].append({
                'timestamp': opportunity.timestamp.isoformat(),
                'profit': opportunity.profit,
                'path': ' → '.join(opportunity.path)
            })
            
            # Keep only the last 100 notifications
            if len(self.notification_history['history']) > 100:
                self.notification_history['history'] = self.notification_history['history'][-100:]
            
            # Save history
            self._save_notification_history()
        
        return success
    
    def _send_email_notification(self, subject, plain_message, html_message):
        """Send an email notification."""
        if not all([self.smtp_server, self.smtp_username, self.smtp_password, self.recipient_email]):
            logger.error("Email notification settings are incomplete")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.smtp_username
            msg['To'] = self.recipient_email
            
            # Attach parts
            part1 = MIMEText(plain_message, 'plain')
            part2 = MIMEText(html_message, 'html')
            msg.attach(part1)
            msg.attach(part2)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
            
            logger.info(f"Email notification sent to {self.recipient_email}")
            return True
        
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
            return False
    
    def _send_telegram_notification(self, message):
        """Send a Telegram notification."""
        if not all([self.telegram_bot_token, self.telegram_chat_id]):
            logger.error("Telegram notification settings are incomplete")
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            data = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, data=data)
            response.raise_for_status()
            
            logger.info(f"Telegram notification sent to chat {self.telegram_chat_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error sending Telegram notification: {e}")
            return False
    
    def _send_discord_notification(self, title, message):
        """Send a Discord notification."""
        if not self.discord_webhook_url:
            logger.error("Discord notification settings are incomplete")
            return False
        
        try:
            data = {
                'embeds': [{
                    'title': title,
                    'description': message,
                    'color': 3066993  # Green color
                }]
            }
            
            response = requests.post(self.discord_webhook_url, json=data)
            response.raise_for_status()
            
            logger.info("Discord notification sent")
            return True
        
        except Exception as e:
            logger.error(f"Error sending Discord notification: {e}")
            return False
    
    def get_notification_stats(self):
        """Get notification statistics."""
        return {
            'enabled': self.enabled,
            'method': self.method,
            'count': self.notification_count,
            'last_time': self.last_notification_time.isoformat() if self.last_notification_time else None,
            'history_count': self.notification_history['count']
        }


# Singleton instance
_notification_instance = None

def get_notification_manager():
    """Get the singleton notification manager instance."""
    global _notification_instance
    if _notification_instance is None:
        _notification_instance = NotificationManager()
    return _notification_instance

if __name__ == "__main__":
    # Test notification
    from datetime import datetime
    
    # Create a test opportunity
    test_opportunity = ArbitrageOpportunity(
        timestamp=datetime.now(),
        start_amount=10000,
        end_amount=10050,
        profit=50,
        profit_percentage=0.5,
        prices={
            'BTC/CAD': 80000.0,
            'ETH/CAD': 5000.0,
            'ETH/BTC': 0.0625
        },
        path=['CAD', 'BTC', 'ETH', 'CAD'],
        is_profitable=True
    )
    
    # Send notification
    notification_manager = get_notification_manager()
    success = notification_manager.send_notification(test_opportunity)
    
    print(f"Notification {'sent successfully' if success else 'failed'}")
    print(f"Method: {notification_manager.method}")
    print(f"Enabled: {notification_manager.enabled}")
    print(f"Count: {notification_manager.notification_count}")