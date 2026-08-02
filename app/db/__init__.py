"""Database package: SQLAlchemy engine, session and ORM models.

The backend is chosen by ``DATABASE_URL``: PostgreSQL in production
(``postgresql+psycopg2://...``), a local SQLite file by default -- so
development and the Render free tier work with zero setup, and the
same code persists to Postgres when the URL is configured.
"""
