import os
from pathlib import Path
import sys

from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.models import Base

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATABASE_URL = f"sqlite:///{(BASE_DIR / 'rag_metadata.db').as_posix()}"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, echo=False, connect_args=connect_args)


def create_tables() -> None:
    Base.metadata.create_all(bind=engine)
    ensure_document_columns()


def ensure_document_columns() -> None:
    inspector = inspect(engine)
    if "documents" not in inspector.get_table_names():
        return

    existing_columns = {
        column["name"]
        for column in inspector.get_columns("documents")
    }
    required_columns = {
        "indexing_status": "VARCHAR NOT NULL DEFAULT 'indexed'",
        "indexed_chunks": "INTEGER NOT NULL DEFAULT 0",
        "indexing_error": "VARCHAR",
    }

    with engine.begin() as connection:
        for column_name, column_definition in required_columns.items():
            if column_name not in existing_columns:
                connection.execute(
                    text(f"ALTER TABLE documents ADD COLUMN {column_name} {column_definition}")
                )


def get_db():
    with Session(bind=engine) as session:
        yield session
