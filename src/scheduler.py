"""Scheduler module for Reddit Radar.

Handles cron-like scheduled analysis runs.
"""

import json
import logging
import threading
import time
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


def parse_cron(cron_expression: str) -> dict:
    """Parse a cron expression into components.

    Supports: minute hour day_of_month month day_of_week
    Example: "0 9 * * *" = daily at 9:00 AM

    Returns:
        Dictionary with 'minute', 'hour', 'day', 'month', 'weekday' keys.
        Values are either integers, lists of integers, or '*' for any.
    """
    parts = cron_expression.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Invalid cron expression: {cron_expression}. Expected 5 fields.")

    def parse_field(field: str, min_val: int, max_val: int) -> list[int] | None:
        """Parse a single cron field. Returns None for '*' (any)."""
        if field == '*':
            return None
        if ',' in field:
            return [int(x) for x in field.split(',')]
        if '-' in field:
            start, end = field.split('-')
            return list(range(int(start), int(end) + 1))
        if '/' in field:
            base, step = field.split('/')
            if base == '*':
                return list(range(min_val, max_val + 1, int(step)))
            else:
                return list(range(int(base), max_val + 1, int(step)))
        return [int(field)]

    return {
        'minute': parse_field(parts[0], 0, 59),
        'hour': parse_field(parts[1], 0, 23),
        'day': parse_field(parts[2], 1, 31),
        'month': parse_field(parts[3], 1, 12),
        'weekday': parse_field(parts[4], 0, 6),  # 0 = Sunday
    }


def get_next_run_time(cron_expression: str, after: datetime | None = None) -> datetime:
    """Calculate the next run time for a cron expression.

    Args:
        cron_expression: Cron expression string.
        after: Calculate next run after this time. Defaults to now.

    Returns:
        Next datetime when the schedule should run.
    """
    if after is None:
        after = datetime.now()

    cron = parse_cron(cron_expression)

    # Start from the next minute
    current = after.replace(second=0, microsecond=0)
    current = current.replace(minute=current.minute + 1) if current.minute < 59 else \
              current.replace(hour=current.hour + 1, minute=0) if current.hour < 23 else \
              current.replace(day=current.day + 1, hour=0, minute=0)

    # Search for the next matching time (max 366 days to prevent infinite loop)
    max_iterations = 366 * 24 * 60  # One year of minutes

    for _ in range(max_iterations):
        # Check if current time matches all cron fields
        matches = True

        if cron['minute'] is not None and current.minute not in cron['minute']:
            matches = False
        if cron['hour'] is not None and current.hour not in cron['hour']:
            matches = False
        if cron['day'] is not None and current.day not in cron['day']:
            matches = False
        if cron['month'] is not None and current.month not in cron['month']:
            matches = False
        if cron['weekday'] is not None:
            # Python weekday: Monday=0, Sunday=6
            # Cron weekday: Sunday=0, Saturday=6
            cron_weekday = (current.weekday() + 1) % 7
            if cron_weekday not in cron['weekday']:
                matches = False

        if matches:
            return current

        # Advance by one minute
        if current.minute < 59:
            current = current.replace(minute=current.minute + 1)
        elif current.hour < 23:
            current = current.replace(hour=current.hour + 1, minute=0)
        else:
            # Next day
            try:
                current = current.replace(day=current.day + 1, hour=0, minute=0)
            except ValueError:
                # End of month, go to next month
                if current.month < 12:
                    current = current.replace(month=current.month + 1, day=1, hour=0, minute=0)
                else:
                    current = current.replace(year=current.year + 1, month=1, day=1, hour=0, minute=0)

    # Fallback: return tomorrow at the first specified hour
    return after.replace(hour=0, minute=0, second=0, microsecond=0) + \
           datetime.timedelta(days=1)


def describe_cron(cron_expression: str) -> str:
    """Generate a human-readable description of a cron expression.

    Args:
        cron_expression: Cron expression string.

    Returns:
        Human-readable description.
    """
    try:
        cron = parse_cron(cron_expression)
    except ValueError:
        return f"Invalid: {cron_expression}"

    parts = []

    # Time
    if cron['minute'] is not None and cron['hour'] is not None:
        if len(cron['minute']) == 1 and len(cron['hour']) == 1:
            hour = cron['hour'][0]
            minute = cron['minute'][0]
            am_pm = "AM" if hour < 12 else "PM"
            display_hour = hour if hour <= 12 else hour - 12
            if display_hour == 0:
                display_hour = 12
            parts.append(f"at {display_hour}:{minute:02d} {am_pm}")
    elif cron['hour'] is not None and len(cron['hour']) == 1:
        hour = cron['hour'][0]
        am_pm = "AM" if hour < 12 else "PM"
        display_hour = hour if hour <= 12 else hour - 12
        if display_hour == 0:
            display_hour = 12
        parts.append(f"at {display_hour}:00 {am_pm}")

    # Day of week
    day_names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    if cron['weekday'] is not None:
        if len(cron['weekday']) == 5 and 1 in cron['weekday'] and 5 in cron['weekday']:
            parts.insert(0, "Weekdays")
        elif len(cron['weekday']) == 2 and 0 in cron['weekday'] and 6 in cron['weekday']:
            parts.insert(0, "Weekends")
        else:
            days = [day_names[d] for d in sorted(cron['weekday'])]
            parts.insert(0, ", ".join(days))
    elif cron['day'] is None and cron['month'] is None:
        parts.insert(0, "Daily")

    if not parts:
        return cron_expression

    return " ".join(parts)


class SchedulerThread(threading.Thread):
    """Background thread that checks for and executes scheduled runs."""

    def __init__(self, check_interval: int = 60):
        """Initialize the scheduler thread.

        Args:
            check_interval: Seconds between schedule checks.
        """
        super().__init__(daemon=True)
        self.check_interval = check_interval
        self._stop_event = threading.Event()
        self._running_jobs: set[str] = set()

    def stop(self):
        """Signal the scheduler to stop."""
        self._stop_event.set()

    def run(self):
        """Main scheduler loop."""
        logger.info("[Scheduler] Starting scheduler thread")

        while not self._stop_event.is_set():
            try:
                self._check_schedules()
            except Exception as e:
                logger.exception(f"[Scheduler] Error checking schedules: {e}")

            # Wait for the next check interval
            self._stop_event.wait(timeout=self.check_interval)

        logger.info("[Scheduler] Scheduler thread stopped")

    def _check_schedules(self):
        """Check for due schedules and execute them."""
        from src.database import get_database

        db = get_database()
        db.initialize()

        due_schedules = db.get_due_schedules()

        for schedule in due_schedules:
            if schedule.schedule_id in self._running_jobs:
                logger.debug(f"[Scheduler] Schedule {schedule.name} is already running, skipping")
                continue

            logger.info(f"[Scheduler] Executing schedule: {schedule.name}")

            # Mark as running
            self._running_jobs.add(schedule.schedule_id)

            # Execute in a separate thread
            thread = threading.Thread(
                target=self._execute_schedule,
                args=(schedule,),
                daemon=True
            )
            thread.start()

    def _execute_schedule(self, schedule):
        """Execute a scheduled analysis."""
        from src.database import get_database, Schedule
        from src.ui.web import create_job, run_analysis_job

        try:
            # Get subreddits
            subreddits = json.loads(schedule.subreddits_json) if schedule.subreddits_json else []

            # If using a preset, load its subreddits
            if schedule.preset_id:
                db = get_database()
                preset = db.get_preset(schedule.preset_id)
                if preset:
                    subreddits = json.loads(preset.subreddits_json)

            if not subreddits:
                logger.warning(f"[Scheduler] No subreddits for schedule {schedule.name}")
                return

            # Create and run the job
            job_id = create_job(
                template=schedule.template,
                subreddits=subreddits,
                posts_per_sub=schedule.posts_per_sub,
                time_window=schedule.time_window_hours,
            )

            logger.info(f"[Scheduler] Started job {job_id} for schedule {schedule.name}")

            # Run the analysis (blocking)
            run_analysis_job(job_id)

            # Send notifications
            self._send_notifications(schedule, job_id)

        except Exception as e:
            logger.exception(f"[Scheduler] Failed to execute schedule {schedule.name}: {e}")
            self._send_failure_notifications(schedule, str(e))
        finally:
            # Update run times
            now = datetime.now().timestamp()
            next_run = get_next_run_time(schedule.cron_expression).timestamp()

            db = get_database()
            db.update_schedule_run_times(schedule.schedule_id, now, next_run)

            # Remove from running set
            self._running_jobs.discard(schedule.schedule_id)

    def _send_notifications(self, schedule, job_id: str):
        """Send notifications for completed run."""
        from src.database import get_database
        from src.ui.web import get_job

        db = get_database()
        notifications = db.get_enabled_notifications("run_complete")

        job = get_job(job_id)
        if not job:
            return

        for notification in notifications:
            try:
                send_notification(notification, {
                    "event": "run_complete",
                    "schedule_name": schedule.name,
                    "job_id": job_id,
                    "run_id": job.get("run_id"),
                    "status": job.get("status"),
                    "timestamp": datetime.now().isoformat(),
                })
            except Exception as e:
                logger.error(f"[Scheduler] Failed to send notification {notification.name}: {e}")

    def _send_failure_notifications(self, schedule, error: str):
        """Send notifications for failed run."""
        from src.database import get_database

        db = get_database()
        notifications = db.get_enabled_notifications("run_failed")

        for notification in notifications:
            try:
                send_notification(notification, {
                    "event": "run_failed",
                    "schedule_name": schedule.name,
                    "error": error,
                    "timestamp": datetime.now().isoformat(),
                })
            except Exception as e:
                logger.error(f"[Scheduler] Failed to send failure notification {notification.name}: {e}")


def send_notification(notification, payload: dict):
    """Send a notification based on its type.

    Args:
        notification: Notification configuration.
        payload: Data to send.
    """
    config = json.loads(notification.config_json)

    if notification.notification_type == "webhook":
        send_webhook(config, payload)
    elif notification.notification_type == "email":
        send_email(config, payload)
    else:
        logger.warning(f"Unknown notification type: {notification.notification_type}")


def send_webhook(config: dict, payload: dict):
    """Send a webhook notification.

    Args:
        config: Webhook configuration with 'url' key.
        payload: Data to send as JSON.
    """
    import httpx

    url = config.get("url")
    if not url:
        logger.error("Webhook URL not configured")
        return

    headers = config.get("headers", {})
    if isinstance(headers, str):
        headers = json.loads(headers)

    headers["Content-Type"] = "application/json"

    try:
        with httpx.Client(timeout=30) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            logger.info(f"[Notification] Webhook sent to {url}: {response.status_code}")
    except Exception as e:
        logger.error(f"[Notification] Webhook failed: {e}")
        raise


def send_email(config: dict, payload: dict):
    """Send an email notification.

    Args:
        config: Email configuration with 'to', 'smtp_host', etc.
        payload: Data to include in email.
    """
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    to_email = config.get("to")
    smtp_host = config.get("smtp_host", "localhost")
    smtp_port = config.get("smtp_port", 587)
    smtp_user = config.get("smtp_user")
    smtp_pass = config.get("smtp_pass")
    from_email = config.get("from", smtp_user or "reddit-radar@localhost")

    if not to_email:
        logger.error("Email 'to' address not configured")
        return

    # Build email
    subject = f"Reddit Radar: {payload.get('event', 'Notification')}"
    body = f"""
Reddit Radar Notification
========================

Event: {payload.get('event')}
Schedule: {payload.get('schedule_name', 'N/A')}
Time: {payload.get('timestamp', 'N/A')}

"""
    if payload.get('run_id'):
        body += f"Run ID: {payload['run_id']}\n"
    if payload.get('error'):
        body += f"Error: {payload['error']}\n"

    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_email
    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            if smtp_user and smtp_pass:
                server.starttls()
                server.login(smtp_user, smtp_pass)
            server.send_message(msg)
            logger.info(f"[Notification] Email sent to {to_email}")
    except Exception as e:
        logger.error(f"[Notification] Email failed: {e}")
        raise


# Global scheduler instance
_scheduler: Optional[SchedulerThread] = None


def start_scheduler(check_interval: int = 60) -> SchedulerThread:
    """Start the global scheduler thread.

    Args:
        check_interval: Seconds between schedule checks.

    Returns:
        The scheduler thread.
    """
    global _scheduler

    if _scheduler is not None and _scheduler.is_alive():
        logger.warning("[Scheduler] Scheduler already running")
        return _scheduler

    _scheduler = SchedulerThread(check_interval=check_interval)
    _scheduler.start()

    return _scheduler


def stop_scheduler():
    """Stop the global scheduler thread."""
    global _scheduler

    if _scheduler is not None:
        _scheduler.stop()
        _scheduler.join(timeout=5)
        _scheduler = None


def get_scheduler() -> Optional[SchedulerThread]:
    """Get the global scheduler instance."""
    return _scheduler
