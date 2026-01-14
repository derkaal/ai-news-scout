"""
Alerting System

Manages alerts, notifications, and escalation for monitoring events.
"""

import logging
import smtplib
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from collections import defaultdict
import threading

from .config import MonitoringConfig


class AlertSeverity:
    """Alert severity levels."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class Alert:
    """Represents an alert."""
    
    def __init__(self, alert_type: str, message: str, severity: str,
                 context: Dict[str, Any] = None):
        self.alert_type = alert_type
        self.message = message
        self.severity = severity
        self.context = context or {}
        self.timestamp = datetime.now()
        self.id = f"{alert_type}_{int(self.timestamp.timestamp())}"
        self.acknowledged = False
        self.resolved = False


class AlertManager:
    """
    Alert management system.
    
    Handles alert generation, notification delivery, cooldown periods,
    and escalation policies.
    """
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Alert tracking
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []
        self._alert_cooldowns: Dict[str, datetime] = {}
        self._alert_counts: Dict[str, int] = defaultdict(int)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Custom handlers
        self._custom_handlers: Dict[str, Callable] = {}
        
        self.logger.info("AlertManager initialized")
    
    def register_custom_handler(self, alert_type: str, handler: Callable):
        """Register custom alert handler."""
        self._custom_handlers[alert_type] = handler
        self.logger.info(f"Registered custom handler for {alert_type}")
    
    def send_alert(self, alert_type: str, message: str, 
                   severity: str = AlertSeverity.WARNING,
                   context: Dict[str, Any] = None) -> bool:
        """
        Send an alert with cooldown and deduplication.
        
        Returns True if alert was sent, False if suppressed.
        """
        with self._lock:
            # Check cooldown
            if self._is_in_cooldown(alert_type):
                self.logger.debug(f"Alert {alert_type} suppressed due to cooldown")
                return False
            
            # Create alert
            alert = Alert(alert_type, message, severity, context)
            
            # Check for duplicate active alerts
            if self._is_duplicate_alert(alert):
                self.logger.debug(f"Duplicate alert {alert_type} suppressed")
                return False
            
            # Add to active alerts
            self._active_alerts[alert.id] = alert
            self._alert_history.append(alert)
            
            # Keep history manageable
            if len(self._alert_history) > 1000:
                self._alert_history = self._alert_history[-1000:]
            
            # Set cooldown
            cooldown_minutes = self.config.alerting.alert_cooldown_minutes
            self._alert_cooldowns[alert_type] = (
                datetime.now() + timedelta(minutes=cooldown_minutes)
            )
            
            # Increment count
            self._alert_counts[alert_type] += 1
            
            self.logger.info(f"Sending {severity} alert [{alert_type}]: {message}")
            
            # Send notifications
            self._send_notifications(alert)
            
            return True
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
        """Acknowledge an alert."""
        with self._lock:
            if alert_id in self._active_alerts:
                alert = self._active_alerts[alert_id]
                alert.acknowledged = True
                alert.context['acknowledged_by'] = acknowledged_by
                alert.context['acknowledged_at'] = datetime.now().isoformat()
                
                self.logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
                return True
            
            return False
    
    def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        """Resolve an alert."""
        with self._lock:
            if alert_id in self._active_alerts:
                alert = self._active_alerts[alert_id]
                alert.resolved = True
                alert.context['resolved_by'] = resolved_by
                alert.context['resolved_at'] = datetime.now().isoformat()
                
                # Remove from active alerts
                del self._active_alerts[alert_id]
                
                self.logger.info(f"Alert {alert_id} resolved by {resolved_by}")
                return True
            
            return False
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts."""
        with self._lock:
            return [
                {
                    'id': alert.id,
                    'type': alert.alert_type,
                    'message': alert.message,
                    'severity': alert.severity,
                    'timestamp': alert.timestamp.isoformat(),
                    'acknowledged': alert.acknowledged,
                    'context': alert.context
                }
                for alert in self._active_alerts.values()
            ]
    
    def get_alert_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert statistics for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            recent_alerts = [
                alert for alert in self._alert_history
                if alert.timestamp >= cutoff_time
            ]
            
            # Count by severity
            severity_counts = defaultdict(int)
            type_counts = defaultdict(int)
            
            for alert in recent_alerts:
                severity_counts[alert.severity] += 1
                type_counts[alert.alert_type] += 1
            
            return {
                'time_period_hours': hours,
                'total_alerts': len(recent_alerts),
                'active_alerts': len(self._active_alerts),
                'by_severity': dict(severity_counts),
                'by_type': dict(type_counts),
                'top_alert_types': sorted(
                    type_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            }
    
    def clear_resolved_alerts(self, older_than_hours: int = 24):
        """Clear resolved alerts older than specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        
        with self._lock:
            original_count = len(self._alert_history)
            self._alert_history = [
                alert for alert in self._alert_history
                if not alert.resolved or alert.timestamp >= cutoff_time
            ]
            
            cleared_count = original_count - len(self._alert_history)
            if cleared_count > 0:
                self.logger.info(f"Cleared {cleared_count} resolved alerts")
    
    def _is_in_cooldown(self, alert_type: str) -> bool:
        """Check if alert type is in cooldown period."""
        if alert_type not in self._alert_cooldowns:
            return False
        
        return datetime.now() < self._alert_cooldowns[alert_type]
    
    def _is_duplicate_alert(self, new_alert: Alert) -> bool:
        """Check if this is a duplicate of an active alert."""
        for active_alert in self._active_alerts.values():
            if (active_alert.alert_type == new_alert.alert_type and
                active_alert.message == new_alert.message and
                not active_alert.resolved):
                return True
        return False
    
    def _send_notifications(self, alert: Alert):
        """Send notifications for an alert."""
        try:
            # Email notifications
            if self.config.alerting.enable_email_alerts:
                self._send_email_notification(alert)
            
            # Slack notifications
            if self.config.alerting.enable_slack_alerts:
                self._send_slack_notification(alert)
            
            # Webhook notifications
            if self.config.alerting.enable_webhook_alerts:
                self._send_webhook_notifications(alert)
            
            # Custom handlers
            if alert.alert_type in self._custom_handlers:
                self._custom_handlers[alert.alert_type](alert)
                
        except Exception as e:
            self.logger.error(f"Error sending notifications for alert {alert.id}: {e}")
    
    def _send_email_notification(self, alert: Alert):
        """Send email notification."""
        try:
            if not self.config.alerting.email_recipients:
                return
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.config.alerting.smtp_username
            msg['To'] = ', '.join(self.config.alerting.email_recipients)
            msg['Subject'] = f"[{alert.severity.upper()}] {alert.alert_type}"
            
            # Email body
            body = f"""
Alert Details:
- Type: {alert.alert_type}
- Severity: {alert.severity}
- Time: {alert.timestamp.isoformat()}
- Message: {alert.message}

Context:
{json.dumps(alert.context, indent=2)}

Service: Newsletter Clustering Agent
Environment: {getattr(self.config, 'environment', 'unknown')}
            """.strip()
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.config.alerting.smtp_server, 
                            self.config.alerting.smtp_port) as server:
                server.starttls()
                server.login(
                    self.config.alerting.smtp_username,
                    self.config.alerting.smtp_password
                )
                server.send_message(msg)
            
            self.logger.info(f"Email notification sent for alert {alert.id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}")
    
    def _send_slack_notification(self, alert: Alert):
        """Send Slack notification."""
        try:
            if not self.config.alerting.slack_webhook_url:
                return
            
            # Determine color based on severity
            color_map = {
                AlertSeverity.CRITICAL: "#ff0000",
                AlertSeverity.WARNING: "#ffaa00",
                AlertSeverity.INFO: "#00aa00"
            }
            color = color_map.get(alert.severity, "#808080")
            
            # Create Slack message
            payload = {
                "channel": self.config.alerting.slack_channel,
                "username": "Newsletter Agent Monitor",
                "icon_emoji": ":warning:",
                "attachments": [
                    {
                        "color": color,
                        "title": f"{alert.severity.upper()}: {alert.alert_type}",
                        "text": alert.message,
                        "fields": [
                            {
                                "title": "Time",
                                "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                                "short": True
                            },
                            {
                                "title": "Service",
                                "value": "Newsletter Clustering Agent",
                                "short": True
                            }
                        ],
                        "footer": f"Alert ID: {alert.id}",
                        "ts": int(alert.timestamp.timestamp())
                    }
                ]
            }
            
            # Add context fields if available
            if alert.context:
                for key, value in alert.context.items():
                    if isinstance(value, (str, int, float)):
                        payload["attachments"][0]["fields"].append({
                            "title": key.replace('_', ' ').title(),
                            "value": str(value),
                            "short": True
                        })
            
            # Send to Slack
            response = requests.post(
                self.config.alerting.slack_webhook_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            
            self.logger.info(f"Slack notification sent for alert {alert.id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send Slack notification: {e}")
    
    def _send_webhook_notifications(self, alert: Alert):
        """Send webhook notifications."""
        for webhook_url in self.config.alerting.webhook_urls:
            try:
                payload = {
                    "alert_id": alert.id,
                    "alert_type": alert.alert_type,
                    "message": alert.message,
                    "severity": alert.severity,
                    "timestamp": alert.timestamp.isoformat(),
                    "context": alert.context,
                    "service": "newsletter-clustering-agent"
                }
                
                response = requests.post(
                    webhook_url,
                    json=payload,
                    timeout=10,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                
                self.logger.info(f"Webhook notification sent to {webhook_url}")
                
            except Exception as e:
                self.logger.error(f"Failed to send webhook notification to {webhook_url}: {e}")


def create_alert_manager(config: MonitoringConfig) -> AlertManager:
    """Create and configure alert manager."""
    return AlertManager(config)