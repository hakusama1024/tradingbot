"""Scheduler for automated daily trading runs."""

import logging
import signal
import sys
from datetime import datetime

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from .orchestrator import Orchestrator
from .social_monitor import SocialFeedMonitor

logger = logging.getLogger(__name__)


class TradingScheduler:
    """Wraps APScheduler to run the orchestrator on a daily schedule."""

    def __init__(self, config: dict):
        self.config = config
        self.orchestrator = Orchestrator(config)
        self.social_monitor = SocialFeedMonitor(config)
        self.scheduler = BlockingScheduler(timezone="US/Eastern")
        self._setup_jobs()
        self._setup_signal_handlers()

    def _setup_jobs(self):
        mode = self.config.get("trading_mode", "swing")

        open_snapshot_time = self.config.get("market_open_snapshot_time", "09:25")
        hour, minute = open_snapshot_time.split(":")
        self.scheduler.add_job(
            self._run_with_logging,
            args=[self.orchestrator.take_market_snapshot, "Market Open Snapshot"],
            trigger=CronTrigger(
                day_of_week="mon-fri",
                hour=int(hour),
                minute=int(minute),
                timezone="US/Eastern",
            ),
            id="market_open_snapshot",
            name="Market Open Snapshot",
            misfire_grace_time=300,
        )
        logger.info(
            "Scheduled market open snapshot at %s ET (Mon-Fri)",
            open_snapshot_time,
        )

        # Swing trade: daily analysis before close
        if mode in ("swing", "both"):
            swing_time = self.config.get("swing_analysis_time", "15:30")
            hour, minute = swing_time.split(":")
            self.scheduler.add_job(
                self._run_with_logging,
                args=[self.orchestrator.run_daily_analysis, "Daily Swing Analysis"],
                trigger=CronTrigger(
                    day_of_week="mon-fri",
                    hour=int(hour),
                    minute=int(minute),
                    timezone="US/Eastern",
                ),
                id="swing_analysis",
                name="Daily Swing Analysis",
                misfire_grace_time=300,
            )
            logger.info(f"Scheduled swing analysis at {swing_time} ET (Mon-Fri)")

        # Day trade: intraday scans
        if mode in ("day", "both"):
            interval = self.config.get("intraday_interval_minutes", 60)
            self.scheduler.add_job(
                self._run_with_logging,
                args=[self.orchestrator.run_daily_analysis, "Intraday Scan"],
                trigger=CronTrigger(
                    day_of_week="mon-fri",
                    hour="10-15",
                    minute=f"*/{interval}" if interval < 60 else "0",
                    timezone="US/Eastern",
                ),
                id="intraday_scan",
                name="Intraday Scan",
                misfire_grace_time=120,
            )
            logger.info(f"Scheduled intraday scans every {interval}min (10AM-3PM ET)")

        # Daily reflection after market close
        reflection_time = self.config.get("reflection_time", "16:30")
        hour, minute = reflection_time.split(":")
        self.scheduler.add_job(
            self._run_with_logging,
            args=[self.orchestrator.run_daily_reflection, "Daily Reflection & Report"],
            trigger=CronTrigger(
                day_of_week="mon-fri",
                hour=int(hour),
                minute=int(minute),
                timezone="US/Eastern",
            ),
            id="daily_reflection",
            name="Daily Reflection & Report",
            misfire_grace_time=600,
        )
        logger.info(f"Scheduled daily reflection at {reflection_time} ET (Mon-Fri)")

        daily_summary_time = self.config.get("daily_summary_time", "16:40")
        hour, minute = daily_summary_time.split(":")
        self.scheduler.add_job(
            self._run_with_logging,
            args=[self.orchestrator.send_daily_notifications, "Daily Summary Push"],
            trigger=CronTrigger(
                day_of_week="mon-fri",
                hour=int(hour),
                minute=int(minute),
                timezone="US/Eastern",
            ),
            id="daily_summary_push",
            name="Daily Summary Push",
            misfire_grace_time=600,
        )
        logger.info(
            "Scheduled daily summary push at %s ET (Mon-Fri)",
            daily_summary_time,
        )

        weekly_summary_time = self.config.get("weekly_summary_time", "16:45")
        hour, minute = weekly_summary_time.split(":")
        self.scheduler.add_job(
            self._run_with_logging,
            args=[self.orchestrator.send_weekly_summary, "Weekly Summary Push"],
            trigger=CronTrigger(
                day_of_week="fri",
                hour=int(hour),
                minute=int(minute),
                timezone="US/Eastern",
            ),
            id="weekly_summary_push",
            name="Weekly Summary Push",
            misfire_grace_time=900,
        )
        logger.info(
            "Scheduled weekly summary push at %s ET (Fri)",
            weekly_summary_time,
        )

        if self.config.get("social_monitor_enabled", False):
            interval = max(int(self.config.get("social_check_interval_minutes", 30)), 1)
            self.scheduler.add_job(
                self._run_with_logging,
                args=[self.social_monitor.check_once, "Social Feed Monitor"],
                trigger=CronTrigger(
                    hour="7-22",
                    minute=f"*/{interval}" if interval < 60 else "0",
                    timezone="US/Eastern",
                ),
                id="social_feed_monitor",
                name="Social Feed Monitor",
                misfire_grace_time=300,
            )
            logger.info(
                "Scheduled social monitor every %smin (7AM-10PM ET)",
                interval,
            )

    def _run_with_logging(self, func, label: str):
        """Execute a job with structured logging."""
        logger.info(f"{'=' * 50}")
        logger.info(f"JOB START: {label} at {datetime.now()}")
        logger.info(f"{'=' * 50}")
        try:
            result = func()
            logger.info(f"JOB DONE: {label} — {result}")
        except Exception as e:
            logger.error(f"JOB FAILED: {label} — {e}", exc_info=True)
            self.orchestrator.notifier.send(
                "TradingAgents Job Failed",
                f"{label}\n{type(e).__name__}: {e}",
                priority="high",
                tags=["warning", "rotating_light"],
                dedupe_key=f"job-failed:{label}:{datetime.now().date().isoformat()}:{type(e).__name__}",
            )

    def _setup_signal_handlers(self):
        """Graceful shutdown on SIGINT/SIGTERM."""
        def _shutdown(signum, frame):
            logger.info("Shutdown signal received. Stopping scheduler...")
            self.scheduler.shutdown(wait=False)
            sys.exit(0)

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

    def start(self):
        """Start the scheduler (blocks forever)."""
        logger.info("=" * 60)
        logger.info("TradingAgents Automation Scheduler Starting")
        logger.info(f"Mode: {self.config.get('trading_mode', 'swing')}")
        logger.info(f"Paper: {self.config.get('paper_trading', True)}")
        logger.info(f"Execution enabled: {self.config.get('execution_enabled', True)}")
        logger.info(f"Watchlist: {self.config.get('watchlist', [])}")
        logger.info("=" * 60)

        # Take initial snapshot
        if self.config.get("execution_enabled", True) and self.orchestrator.tracker is not None:
            try:
                self.orchestrator.tracker.take_daily_snapshot()
            except Exception as e:
                logger.warning(f"Could not take initial snapshot: {e}")

        # Print upcoming jobs
        jobs = self.scheduler.get_jobs()
        for job in jobs:
            next_run = getattr(job, "next_run_time", None)
            if next_run is not None:
                logger.info(f"  Job: {job.name} -> next run: {next_run}")
            else:
                logger.info(f"  Job: {job.name} -> trigger: {job.trigger}")

        self.scheduler.start()

    def run_now(self):
        """Immediately run analysis (for manual/testing use)."""
        logger.info("Manual trigger: running analysis NOW")
        return self.orchestrator.run_daily_analysis()
