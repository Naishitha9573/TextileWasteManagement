"""Safe SQLite -> PostgreSQL migration for the Textile Waste platform.

Reads every application table from a legacy SQLite database file and copies
the records into the primary PostgreSQL database, preserving primary keys,
timestamps and relationships.

Safety properties:
- Never drops or truncates anything.
- Backs up the SQLite file before touching data.
- Idempotent: rows whose primary key already exists in the target are skipped,
  so the script can be rerun safely.
- Reports per-table migrated/skipped/total counts and verifies final row
  counts on both sides.
- Repairs PostgreSQL identity sequences after explicit-ID inserts.

Usage:
    python scripts/migrate_sqlite_to_postgres.py --source ./textile_waste.db
    python scripts/migrate_sqlite_to_postgres.py --source legacy.db \
        --target postgresql+psycopg2://user:pass@localhost:5432/textile_waste
    python scripts/migrate_sqlite_to_postgres.py --source legacy.db --verify-only
"""

import argparse
import datetime
import shutil
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from sqlalchemy import create_engine, func, inspect as sql_inspect  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

# Importing database.py resolves DATABASE_URL (PostgreSQL by default) and may
# raise a clear RuntimeError if PostgreSQL is unreachable.
import database  # noqa: E402
from database import (  # noqa: E402
    AnalysisResult,
    Notification,
    User,
    WasteBatch,
)

# Insert order respects foreign keys: users -> batches -> analyses -> notifications
TABLES = [
    ("users", User),
    ("waste_batches", WasteBatch),
    ("analysis_results", AnalysisResult),
    ("notifications", Notification),
]


def make_backup(sqlite_path: Path) -> Path:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = sqlite_path.with_name(
        f"{sqlite_path.stem}_backup_{timestamp}{sqlite_path.suffix}"
    )
    shutil.copy2(sqlite_path, backup_path)
    print(f"[backup] SQLite backed up to: {backup_path}")
    return backup_path


def open_sessions(source_url: str, target_engine):
    source_engine = create_engine(source_url)
    SourceSession = sessionmaker(bind=source_engine)
    TargetSession = sessionmaker(bind=target_engine)
    return SourceSession(), TargetSession()


def ensure_target_schema(target_engine):
    database.Base.metadata.create_all(bind=target_engine)
    print("[schema] target schema ensured (create_all, non-destructive)")


def table_counts(session):
    return {name: session.query(func.count(model.id)).scalar() for name, model in TABLES}


def migrate_table(source_session, target_session, model):
    """Copy missing rows for one model. Returns (inserted, skipped)."""
    existing_ids = {row[0] for row in target_session.query(model.id).all()}
    inserted = skipped = 0
    for row in source_session.query(model).all():
        if row.id in existing_ids:
            skipped += 1
            continue
        target_session.merge(row)  # merge keeps explicit PK values
        existing_ids.add(row.id)
        inserted += 1
    target_session.commit()
    return inserted, skipped


def fix_sequences(target_engine):
    """After inserting explicit PKs, resync PostgreSQL identity sequences."""
    if target_engine.dialect.name != "postgresql":
        return
    from sqlalchemy import text

    with target_engine.begin() as conn:
        for table, model in TABLES:
            pk_col = model.__mapper__.primary_key[0].name
            seq = f"{table}_{pk_col}_seq"
            conn.execute(
                text(
                    f"SELECT setval('{seq}', COALESCE((SELECT MAX({pk_col}) "
                    f"FROM {table}), 1), true)"
                )
            )
    print("[sequences] PostgreSQL identity sequences resynced")


def verify(source_counts, target_counts):
    print("\n=== VERIFICATION ===")
    ok = True
    for name, _ in TABLES:
        s, t = source_counts.get(name, 0), target_counts.get(name, 0)
        status = "OK" if t >= s else "MISMATCH"
        if t < s:
            ok = False
        print(f"{name:20s} sqlite={s:6d}  postgresql={t:6d}  [{status}]")
    return ok


def main():
    parser = argparse.ArgumentParser(description="SQLite -> PostgreSQL migration")
    parser.add_argument(
        "--source",
        required=True,
        help="Path to the legacy SQLite .db file",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Target SQLAlchemy URL (defaults to app DATABASE_URL / PostgreSQL)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip the automatic SQLite backup (not recommended)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only report row counts on both sides; copy nothing",
    )
    args = parser.parse_args()

    sqlite_path = Path(args.source)
    if not sqlite_path.exists():
        print(f"No SQLite database found at '{sqlite_path}'. Nothing to migrate.")
        return 0

    source_url = f"sqlite:///{sqlite_path.as_posix()}"
    target_engine = (
        create_engine(args.target) if args.target else database.engine
    )
    try:
        with target_engine.connect():
            pass
    except Exception as exc:
        print(f"FATAL: cannot reach target database: {exc}")
        return 1

    source_session, target_session = open_sessions(source_url, target_engine)
    try:
        source_counts = table_counts(source_session)
        total_rows = sum(source_counts.values())
        print(f"[source] {sqlite_path}: {total_rows} total rows across {len(TABLES)} tables")
        for name, count in source_counts.items():
            print(f"         {name:20s} {count}")

        if args.verify_only:
            ensure_target_schema(target_engine)
            target_counts = table_counts(target_session)
            return 0 if verify(source_counts, target_counts) else 1

        if not args.no_backup:
            make_backup(sqlite_path)

        ensure_target_schema(target_engine)

        print("\n=== MIGRATION ===")
        any_failure = False
        for name, model in TABLES:
            try:
                inserted, skipped = migrate_table(source_session, target_session, model)
                print(f"{name:20s} inserted={inserted:6d} skipped(already present)={skipped:6d}")
            except Exception as exc:
                any_failure = True
                target_session.rollback()
                print(f"{name:20s} FAILED: {exc}")
        if any_failure:
            print("\nSome tables failed to migrate; see errors above. "
                  "The script is safe to rerun after fixing the issue.")
            return 1

        fix_sequences(target_engine)

        target_counts = table_counts(target_session)
        success = verify(source_counts, target_counts)
        print(
            "\nMigration complete."
            if success
            else "\nWARNING: target counts are lower than source — investigate before proceeding."
        )
        return 0 if success else 1
    finally:
        source_session.close()
        target_session.close()


if __name__ == "__main__":
    raise SystemExit(main())
