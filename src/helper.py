import sqlite3
import re
import logging
from typing import Dict, List, Optional, Tuple, Union
from contextlib import contextmanager
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

def validate_sql(sql: str) -> Tuple[bool, str]:
    """
    Basic SQL query validation for structure.
    Returns: (is_valid, message)
    """
    if not sql.strip():
        return False, "Empty SQL query"
    
    # Check for basic SQL structure
    if not re.search(r'SELECT.*FROM', sql, re.IGNORECASE):
        return False, "Missing SELECT...FROM"
    
    return True, "Valid SQL"

def preprocess_query(query: str) -> str:
    """
    Clean and normalize natural language or SQL query.
    Removes extra whitespace and normalizes formatting.
    """
    query = query.strip()
    query = re.sub(r'\s+', ' ', query)
    return query

def clean_sql_output(sql: str) -> str:
    """
    Clean and format SQL query.
    Ensures consistent formatting and proper semicolon termination.
    """
    sql = sql.strip()
    sql = re.sub(r'(--.*|\s+|\n+)', ' ', sql)
    sql = re.sub(r'\s+', ' ', sql)
    if not sql.endswith(';'):
        sql += ';'
    return sql

def log_training_metrics(metrics: Dict[str, Union[float, int, str]], prefix: str = "train") -> None:
    """
    Log training metrics in a consistent format.
    Args:
        metrics: Dictionary of metric names and values
        prefix: Prefix for the log message (e.g., 'train' or 'eval')
    """
    log_msg = f"[{prefix.upper()}] " + " | ".join(
        f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
        for k, v in metrics.items()
    )
    logger.info(log_msg)

def timestamp() -> str:
    """
    Generate consistent timestamp format for logging and file naming.
    Returns: Formatted timestamp string
    """
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.
    Args:
        name: Name for the logger
    Returns: Configured logger instance
    """
    return logging.getLogger(name)