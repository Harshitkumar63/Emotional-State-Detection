"""
Dashboard Analytics — Aggregated Burnout Statistics & Visualisations
=====================================================================
Computes and formats analytics data from the SessionStore for the
enterprise dashboard.

Provides:
  - Risk distribution over time
  - Stress/energy trend lines
  - Weekly heatmap data
  - CSV/JSON report generation
  - Escalation alerts

Why a separate analytics module?
  Separating analytics from storage (SessionStore) and presentation
  (Streamlit) follows separation of concerns.  The analytics module
  can be unit-tested independently and reused in different frontends.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime
from typing import Optional

from src.temporal.session_store import SessionStore
from src.utils.helpers import setup_logging

logger = setup_logging()


class DashboardAnalytics:
    """Compute analytics and generate reports from assessment history."""

    def __init__(self, store: SessionStore):
        self.store = store

    # ------------------------------------------------------------------
    # Summary Statistics
    # ------------------------------------------------------------------

    def get_overview(self, user_id: str = "default", days: int = 30) -> dict:
        """Get a comprehensive dashboard overview.

        Returns
        -------
        dict with:
            stats           - aggregate statistics
            trend_data      - time-series for line charts
            risk_distribution - for pie/bar charts
            alerts          - escalation warnings
        """
        stats = self.store.get_statistics(user_id=user_id, days=days)
        trend_data = self.store.get_trend_data(user_id=user_id, days=days)
        alerts = self._check_alerts(stats, trend_data)

        return {
            "stats": stats,
            "trend_data": trend_data,
            "risk_distribution": stats.get("risk_distribution", {}),
            "alerts": alerts,
        }

    # ------------------------------------------------------------------
    # Alert System
    # ------------------------------------------------------------------

    @staticmethod
    def _check_alerts(stats: dict, trend_data: list) -> list[dict]:
        """Generate alerts based on concerning patterns.

        Returns a list of alert dicts with severity, message, and suggestion.
        """
        alerts = []

        # Check for escalation
        if stats.get("escalation_detected"):
            alerts.append({
                "severity": "high",
                "message": "Stress levels have been increasing over recent assessments.",
                "suggestion": (
                    "Consider taking a break, talking to someone you trust, "
                    "or consulting a wellness professional."
                ),
            })

        # Check for persistent high stress
        if stats.get("avg_stress", 0) > 0.7:
            alerts.append({
                "severity": "high",
                "message": "Average stress level is consistently high.",
                "suggestion": (
                    "Persistent high stress is a key burnout indicator. "
                    "Please prioritise self-care and consider professional support."
                ),
            })

        # Check for low energy
        if stats.get("avg_energy", 1) < 0.3:
            alerts.append({
                "severity": "medium",
                "message": "Average energy level is low.",
                "suggestion": (
                    "Low energy over time can signal exhaustion. "
                    "Ensure adequate sleep, nutrition, and breaks."
                ),
            })

        # Check for high-risk frequency
        risk_dist = stats.get("risk_distribution", {})
        total = sum(risk_dist.values())
        if total > 0:
            high_risk_pct = risk_dist.get("High Risk", 0) / total
            if high_risk_pct > 0.5:
                alerts.append({
                    "severity": "high",
                    "message": (
                        f"Over {high_risk_pct:.0%} of recent assessments "
                        "flagged High Risk."
                    ),
                    "suggestion": (
                        "This pattern is concerning. Please consider "
                        "speaking with a mental health professional."
                    ),
                })

        if not alerts:
            alerts.append({
                "severity": "info",
                "message": "No concerning patterns detected.",
                "suggestion": "Keep monitoring and maintain healthy habits.",
            })

        return alerts

    # ------------------------------------------------------------------
    # Export: CSV
    # ------------------------------------------------------------------

    def export_csv(self, user_id: str = "default", days: int = 30) -> str:
        """Export assessment history as a CSV string.

        Returns
        -------
        str : CSV-formatted data ready to download.
        """
        records = self.store.get_history(user_id=user_id, days=days, limit=9999)

        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "Timestamp", "Burnout Risk", "Confidence",
            "Primary Emotion", "Energy Score", "Stress Score",
            "Work Score", "Modalities Used",
        ])

        # Data rows
        for r in records:
            mods = r.get("modalities_used", [])
            if isinstance(mods, list):
                mods = ", ".join(mods)
            writer.writerow([
                r.get("timestamp", ""),
                r.get("burnout_risk", ""),
                f"{r.get('burnout_confidence', 0):.2%}",
                r.get("primary_emotion", ""),
                f"{r.get('energy_score', 0):.3f}",
                f"{r.get('stress_score', 0):.3f}",
                f"{r.get('work_score', 0):.3f}",
                mods,
            ])

        return output.getvalue()

    # ------------------------------------------------------------------
    # Export: JSON Report
    # ------------------------------------------------------------------

    def export_report(self, user_id: str = "default", days: int = 30) -> str:
        """Generate a comprehensive JSON report.

        Includes statistics, trend data, alerts, and full history.
        """
        overview = self.get_overview(user_id=user_id, days=days)
        history = self.store.get_history(user_id=user_id, days=days, limit=9999)

        report = {
            "report_generated": datetime.now().isoformat(),
            "period_days": days,
            "user_id": user_id,
            "summary": overview["stats"],
            "alerts": overview["alerts"],
            "assessment_history": [
                {
                    "timestamp": r.get("timestamp"),
                    "burnout_risk": r.get("burnout_risk"),
                    "primary_emotion": r.get("primary_emotion"),
                    "energy_score": r.get("energy_score"),
                    "stress_score": r.get("stress_score"),
                    "work_score": r.get("work_score"),
                }
                for r in history
            ],
            "disclaimer": (
                "This report is generated by an automated system for "
                "self-awareness purposes only. It is not a medical or "
                "clinical assessment. Please consult a qualified "
                "professional for health-related concerns."
            ),
        }

        return json.dumps(report, indent=2, ensure_ascii=False)
