"""
db/models.py
SQLAlchemy models + DB engine setup.
Supports SQLite (default, zero-config) and PostgreSQL (set DATABASE_URL in .env).
"""
import os
from datetime import datetime

from dotenv import load_dotenv
from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer, String, Text, create_engine, text
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./warranty_fraud.db")

# SQLite needs check_same_thread=False; PostgreSQL ignores it
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


# ── Tables ────────────────────────────────────────────────────────────────────

class WarrantyRegistration(Base):
    __tablename__ = "warranty_registrations"

    id             = Column(Integer, primary_key=True, index=True)
    # User details
    name           = Column(String(255), nullable=False)
    mobile         = Column(String(20),  nullable=False)
    email          = Column(String(255), nullable=False)
    allow_contact  = Column(Boolean, default=False)
    accept_marketing = Column(Boolean, default=False)
    # Product details
    product_name   = Column(String(500), nullable=False)
    marketplace    = Column(String(100), nullable=False)
    purchase_date  = Column(String(50),  nullable=True)
    # Invoice details extracted by OCR
    invoice_number = Column(String(255), nullable=True, index=True)
    ocr_product    = Column(String(500), nullable=True)
    ocr_seller     = Column(String(255), nullable=True)
    ocr_amount     = Column(String(50),  nullable=True)
    # Risk & decision
    risk_score     = Column(Float,   default=0.0)
    decision       = Column(String(20),  default="PENDING")   # APPROVE / MANUAL_REVIEW / REJECT
    decision_reason= Column(Text,    nullable=True)
    # Flags
    is_duplicate   = Column(Boolean, default=False)
    ela_risk       = Column(Float,   default=0.0)
    metadata_risk  = Column(Float,   default=0.0)
    product_risk   = Column(Float,   default=0.0)
    seller_risk    = Column(Float,   default=0.0)
    date_risk      = Column(Float,   default=0.0)
    # Timestamps
    created_at     = Column(DateTime, default=datetime.utcnow)
    updated_at     = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AuthorizedSeller(Base):
    __tablename__ = "authorized_sellers"

    id          = Column(Integer, primary_key=True, index=True)
    seller_name = Column(String(255), nullable=False)
    marketplace = Column(String(100), nullable=False)
    status      = Column(String(20), default="active")   # active / inactive


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_db():
    """FastAPI dependency — yields a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create tables and seed authorized sellers."""
    Base.metadata.create_all(bind=engine)

    # Ensure allow_contact and accept_marketing columns exist
    with engine.begin() as conn:
        try:
            cursor = conn.execute(text("PRAGMA table_info(warranty_registrations)"))
            columns = [row[1] for row in cursor.fetchall()]
            if "allow_contact" not in columns:
                conn.execute(text("ALTER TABLE warranty_registrations ADD COLUMN allow_contact BOOLEAN DEFAULT 0"))
            if "accept_marketing" not in columns:
                conn.execute(text("ALTER TABLE warranty_registrations ADD COLUMN accept_marketing BOOLEAN DEFAULT 0"))
        except Exception:
            pass

    db = SessionLocal()
    try:
        seed_data = [
            ("RetailNet",       "Amazon"),
            ("Ambrane Official","Amazon"),
            ("Boat Official",   "Flipkart"),
            ("Noise Official",  "Amazon"),
            ("Zebronics",       "Flipkart"),
            ("Croma",           "Croma"),
            ("Reliance Digital", "Reliance Digital"),
            ("BLINK COMMERCE PRIVATE LIMITED", "Blinkit"),
        ]
        for seller_name, marketplace in seed_data:
            existing = (
                db.query(AuthorizedSeller)
                .filter(
                    AuthorizedSeller.seller_name == seller_name,
                    AuthorizedSeller.marketplace.ilike(marketplace)
                )
                .first()
            )
            if not existing:
                db.add(AuthorizedSeller(seller_name=seller_name, marketplace=marketplace))
        db.commit()
    finally:
        db.close()