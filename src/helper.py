import sqlite3
import re
import logging
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def execute_query(self, query: str) -> List[Dict]:
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            self.logger.error(f"Database error: {e}")
            raise

def validate_sql(sql: str) -> Tuple[bool, Optional[str]]:
    blacklist = ["drop", "delete", "truncate", "update", "insert", ";", "--"]
    sql = sql.lower()
    if any(word in sql for word in blacklist):
        return False, f"Forbidden operation detected"
    return True, None

def preprocess_query(query: str) -> str:
    query = re.sub(r'\s+', ' ', query.strip()).lower()
    return re.sub(r'[^\w\s.,?]', '', query)

def clean_sql_output(sql: str) -> str:
    sql = re.sub(r'(--.*|\s+|\n+)', ' ', sql).strip()
    return re.sub(r'\s+', ' ', sql).replace(' ;', ';')

def log_training_metrics(metrics: Dict, prefix: str = "train") -> None:
    logger = logging.getLogger(__name__)
    log_msg = f"[{prefix.upper()}] " + " | ".join(
        f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
        for k, v in metrics.items()
    )
    logger.info(log_msg)

def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")