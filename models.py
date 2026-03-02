from sqlmodel import Field, SQLModel
from pydantic import BaseModel
from uuid import UUID, uuid4


class Transaction(SQLModel, table=True):
    id: UUID | None = Field(default_factory=uuid4, primary_key=True)
    username: str | None = Field(foreign_key="user.username", index=True)
    amount: int | None = Field(index=True)
    title: str | None = Field(index = True)
    memo: str | None = Field(default=None, index=True)
    account_name: str | None = Field(default=None, index=True)
    category: str | None = Field(default=None, index=True)
    date: str | None = Field(default=None, index=True)


class Pot(SQLModel, table=True):
    pot_id: UUID | None = Field(default_factory=uuid4, primary_key=True)
    username: str | None = Field(foreign_key="user.username", index=True)
    title: str | None = Field(index=True)
    target_amount: int | None = Field(index=True)
    amount: int | None = Field(default_factory=lambda: 0, index=True)


class PotItem(SQLModel, table=True):
    pot_item_id: UUID | None = Field(default_factory=uuid4, primary_key=True)
    username: str | None = Field(foreign_key="user.username", index=True)
    pot_id: UUID = Field(foreign_key="pot.pot_id", index=True)
    amount: int | None = Field(index=True)
    title: str | None = Field(index=True)
    transfer_from: str | None = Field(index=True)
    transfer_to: str | None = Field(index=True)
    date: str | None = Field(default=None, index=True)

class UserCreate(BaseModel):
    username: str
    password: str

class User(SQLModel, table=True):
    username: str = Field(primary_key=True)
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str | None = None