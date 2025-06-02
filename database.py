# database.py
from sqlmodel import create_engine, SQLModel, Session
from pgvector.sqlalchemy import Vector
import os

# Your PostgreSQL connection URL from the request
# For production, use environment variables!
DATABASE_URL = "postgresql+psycopg2://postgres:DBSB3272@localhost/face" # Using psycopg2 for synchronous operations

# Using psycopg2 driver for synchronous operations
engine = create_engine(DATABASE_URL, echo=True)

def create_db_and_tables():
    # This creates tables synchronously
    SQLModel.metadata.create_all(engine)

# Dependency to get a session
def get_session():
    with Session(engine) as session:
        yield session