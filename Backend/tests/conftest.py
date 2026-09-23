"""Pytest configuration.

The application's primary database is PostgreSQL. For tests we use an
EXPLICIT, isolated SQLite database file so the suite never touches a real
PostgreSQL instance. Set TEST_DATABASE_URL to run the suite against
PostgreSQL instead, e.g.:

    set TEST_DATABASE_URL=postgresql+psycopg2://postgres:pass@localhost:5432/textile_waste_test

This must run before any application module is imported.
"""

import os
import sys

import pytest

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

TEST_DB_FILE = os.path.join(BACKEND_DIR, "test_textile_waste.db")
TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", f"sqlite:///{TEST_DB_FILE}")

os.environ["DATABASE_URL"] = TEST_DATABASE_URL


@pytest.fixture(scope="session", autouse=True)
def _isolated_test_database():
    """Remove any stale test database file before the session starts."""
    if TEST_DATABASE_URL.startswith("sqlite") and os.path.exists(TEST_DB_FILE):
        try:
            os.remove(TEST_DB_FILE)
        except OSError:
            pass
    yield
    if TEST_DATABASE_URL.startswith("sqlite") and os.path.exists(TEST_DB_FILE):
        try:
            os.remove(TEST_DB_FILE)
        except OSError:
            pass
