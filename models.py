# models.py
from typing import List, Optional
from sqlmodel import Field, Relationship, SQLModel
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column


class Users(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)

    # Relationship to link a user to their face encodings
    face_encodings: List["FaceEncoding"] = Relationship(back_populates="user")


class FaceEncoding(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # The 128-dimensional face embedding
    embedding: List[float] = Field(sa_column=Column(Vector(128)))

    user_id: Optional[int] = Field(default=None, foreign_key="users.id")
    user: Optional[Users] = Relationship(back_populates="face_encodings")