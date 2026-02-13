"""
Session Store — Persistent Assessment History via SQLite
=========================================================
Stores burnout assessment results in a local SQLite database so that:

1. History survives browser refreshes and app restarts
2. Temporal trend analysis can look at days/weeks of data
3. The enterprise dashboard can aggregate across sessions

Why SQLite?
  - Zero configuration, no server required
  - Built into Python's standard library (no extra dependency)
  - ACID-compliant (data integrity guaranteed)
  - Perfect for single-user local applications
  - Easy to export or migrate later

Privacy:
  - Database is stored locally on the user's machine
  - No network calls, no cloud sync
  - User can delete the DB file to erase all history
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from src.utils.helpers import setup_logging

logger = setup_logging()

_DEFAULT_DB_PATH = "data/session_history.db"


class SessionStore:
    """Persistent storage for burnout assessment history."""

    def __init__(self, db_path: str = _DEFAULT_DB_PATH):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info("Session store ready: %s", db_path)

    # ------------------------------------------------------------------
    # Database setup
    # ------------------------------------------------------------------

    def _init_db(self):
        """Create the assessments table if it doesn't exist."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS assessments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    user_id TEXT DEFAULT 'default',
                    burnout_risk TEXT,
                    burnout_confidence REAL,
                    primary_emotion TEXT,
                    energy_score REAL,
                    stress_score REAL,
                    work_score REAL,
                    modalities_used TEXT,
                    emotion_scores TEXT,
                    burnout_probabilities TEXT,
                    modality_contributions TEXT,
                    full_state TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON assessments(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user
                ON assessments(user_id)
            """)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def save_assessment(self, state_dict: dict, user_id: str = "default") -> int:
        """Store a completed assessment.

        Parameters
        ----------
        state_dict : dict
            Output of EmotionalState.to_dict().
        user_id : str
            Identifier for multi-user scenarios (default: 'default').

        Returns
        -------
        int : the row ID of the inserted record.
        """
        with self._connect() as conn:
            cursor = conn.execute("""
                INSERT INTO assessments (
                    timestamp, user_id, burnout_risk, burnout_confidence,
                    primary_emotion, energy_score, stress_score, work_score,
                    modalities_used, emotion_scores, burnout_probabilities,
                    modality_contributions, full_state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                user_id,
                state_dict.get("burnout_risk", "N/A"),
                state_dict.get("burnout_confidence", 0.0),
                state_dict.get("primary_emotion", "neutral"),
                state_dict.get("energy_score", 0.5),
                state_dict.get("stress_score", 0.5),
                state_dict.get("work_score", 0.5),
                json.dumps(state_dict.get("modalities_used", [])),
                json.dumps(state_dict.get("emotion_scores", {})),
                json.dumps(state_dict.get("burnout_probabilities", {})),
                json.dumps(state_dict.get("modality_contributions", {})),
                json.dumps(state_dict),
            ))
            row_id = cursor.lastrowid
            logger.info("Assessment saved (id=%d, risk=%s)",
                        row_id, state_dict.get("burnout_risk"))
            return row_id

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_history(
        self,
        user_id: str = "default",
        limit: int = 100,
        days: Optional[int] = None,
    ) -> list[dict]:
        """Retrieve assessment history, newest first.

        Parameters
        ----------
        user_id : str
            Filter by user.
        limit : int
            Maximum records to return.
        days : int, optional
            Only return records from the last N days.

        Returns
        -------
        list of dicts with assessment data.
        """
        query = "SELECT * FROM assessments WHERE user_id = ?"
        params: list = [user_id]

        if days is not None:
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            query += " AND timestamp >= ?"
            params.append(cutoff)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_dict(row) for row in rows]

    def get_trend_data(
        self,
        user_id: str = "default",
        days: int = 30,
    ) -> list[dict]:
        """Get time-series data for trend charts.

        Returns a list of dicts with timestamp and key scores,
        ordered chronologically (oldest first).
        """
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT timestamp, burnout_risk, burnout_confidence,
                       energy_score, stress_score, work_score,
                       primary_emotion
                FROM assessments
                WHERE user_id = ? AND timestamp >= ?
                ORDER BY timestamp ASC
            """, (user_id, cutoff)).fetchall()

        return [dict(row) for row in rows]

    def get_statistics(self, user_id: str = "default", days: int = 30) -> dict:
        """Compute aggregate statistics for the dashboard.

        Returns
        -------
        dict with:
            total_assessments, risk_distribution, avg_energy, avg_stress,
            avg_work, most_common_emotion, escalation_detected
        """
        records = self.get_history(user_id=user_id, days=days, limit=9999)

        if not records:
            return {
                "total_assessments": 0,
                "risk_distribution": {},
                "avg_energy": 0.0,
                "avg_stress": 0.0,
                "avg_work": 0.0,
                "most_common_emotion": "N/A",
                "escalation_detected": False,
            }

        # Risk distribution
        risk_counts = {}
        emotions = []
        energy_sum = stress_sum = work_sum = 0.0

        for r in records:
            risk = r.get("burnout_risk", "N/A")
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
            emotions.append(r.get("primary_emotion", "neutral"))
            energy_sum += r.get("energy_score", 0.5)
            stress_sum += r.get("stress_score", 0.5)
            work_sum += r.get("work_score", 0.5)

        n = len(records)

        # Most common emotion
        from collections import Counter
        emotion_counts = Counter(emotions)
        most_common = emotion_counts.most_common(1)[0][0]

        # Escalation detection: are recent assessments worse than older ones?
        escalation = False
        if n >= 4:
            recent = records[:n // 3]  # newest third
            older = records[2 * n // 3:]  # oldest third
            recent_stress = sum(r.get("stress_score", 0) for r in recent) / len(recent)
            older_stress = sum(r.get("stress_score", 0) for r in older) / len(older)
            if recent_stress > older_stress + 0.15:
                escalation = True

        return {
            "total_assessments": n,
            "risk_distribution": risk_counts,
            "avg_energy": round(energy_sum / n, 3),
            "avg_stress": round(stress_sum / n, 3),
            "avg_work": round(work_sum / n, 3),
            "most_common_emotion": most_common,
            "escalation_detected": escalation,
        }

    # ------------------------------------------------------------------
    # Temporal sequence (for GRU model)
    # ------------------------------------------------------------------

    def get_feature_sequence(
        self,
        user_id: str = "default",
        seq_length: int = 10,
    ) -> list[list[float]]:
        """Get the last N assessments as a feature vector sequence.

        Each assessment is encoded as a 10-d feature vector:
          [energy, stress, work, burnout_risk_encoded,
           anger, disgust, fear, joy, neutral, sadness]

        This is the input format for the GRU temporal model.
        """
        records = self.get_history(user_id=user_id, limit=seq_length)
        records.reverse()  # oldest first for sequence input

        risk_encoding = {"Low Risk": 0.0, "Moderate Risk": 0.5, "High Risk": 1.0}

        sequence = []
        for r in records:
            emotions = r.get("emotion_scores", {})
            if isinstance(emotions, str):
                emotions = json.loads(emotions)

            vec = [
                r.get("energy_score", 0.5),
                r.get("stress_score", 0.5),
                r.get("work_score", 0.5),
                risk_encoding.get(r.get("burnout_risk", "N/A"), 0.5),
                emotions.get("anger", 0.0),
                emotions.get("disgust", 0.0),
                emotions.get("fear", 0.0),
                emotions.get("joy", 0.0),
                emotions.get("neutral", 0.0),
                emotions.get("sadness", 0.0),
            ]
            sequence.append(vec)

        return sequence

    # ------------------------------------------------------------------
    # Admin
    # ------------------------------------------------------------------

    def clear_history(self, user_id: str = "default"):
        """Delete all assessments for a user."""
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM assessments WHERE user_id = ?", (user_id,)
            )
        logger.info("History cleared for user: %s", user_id)

    def count(self, user_id: str = "default") -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM assessments WHERE user_id = ?",
                (user_id,)
            ).fetchone()
            return row[0] if row else 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        """Convert a sqlite3.Row to a plain dict with parsed JSON fields."""
        d = dict(row)
        for key in ("modalities_used", "emotion_scores",
                     "burnout_probabilities", "modality_contributions"):
            if key in d and isinstance(d[key], str):
                try:
                    d[key] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d
