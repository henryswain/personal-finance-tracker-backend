# from fastapi import FastAPI, HTTPException, Depends
# from pydantic import BaseModel
# from typing import Annotated, List
# import models
# from database import engine, SessionLocal
# from sqlalchemy.orm import Session

# app = FastAPI()
# models.Base.metadata.creat_all(bind=engine)

# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close

# db_dependancy = Annotated[Session, Depends(get_db)]



from typing import Annotated, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from sqlmodel import Field, Session, SQLModel, create_engine, select
import os
from uuid import UUID, uuid4
from pydantic import BaseModel
from dotenv import load_dotenv
from contextlib import asynccontextmanager

from fastapi.middleware.cors import CORSMiddleware


load_dotenv()
URL = os.getenv('URL_DATABASE')

engine = create_engine(URL)

# sqlite_file_name = "database.db"
# sqlite_url = f"sqlite:///{sqlite_file_name}"

connect_args = {"check_same_thread": False}
engine = create_engine(URL)

class Transaction(SQLModel, table=True):
    id: UUID | None = Field(default_factory=uuid4, primary_key=True)
    amount: float | None = Field(index=True)
    title: str | None = Field(index = True)
    memo: str | None = Field(default=None, index=True)
    account_name: str | None = Field(default=None, index=True)
    category: str | None = Field(default=None, index=True)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("starting fastapi app")
    create_db_and_tables()
    yield
    print("stopping fastapi app")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Update(BaseModel):
    amount: int | None = None
    title: str | None = None
    memo: str | None = None
    account_name: str | None = None
    transfer_from: str | None = None
    category: str | None = None

# @app.on_event("startup")
# def on_startup():
#     create_db_and_tables()


@app.post("/transaction")
def create_transaction(transaction: Transaction, session: SessionDep) -> dict:
    try:
        session.add(transaction)
    except Exception as err:
        print(f"Unexpected {err=} trying session.add(transaction), {type(err)=}")
        raise
    session.commit()
    session.refresh(transaction)
    return {"status": 200, "data": transaction}

# @app.put("/transactions/{transaction_id}")
# def update_transaction(session: SessionDep, transaction_id: int, updates: Update) -> Transaction:
#     statement = select(Transaction).where(Transaction.id == transaction_id)
#     result = session.exec(statement)
#     for key in updates:
#         result[key] = updates[key]
#     session.add(result)
#     session.commit()
#     session.refresh(result)
#     return result

@app.patch("/transactions/{transaction_id}")
def update_transaction(transaction_id: UUID, transaction: Update) -> dict:
    with Session(engine) as session:
        db_transaction = session.get(Transaction, transaction_id)
        if not db_transaction:
            raise HTTPException(status_code=404, detail="Hero not found")
        transaction_data = transaction.model_dump(exclude_unset=True)
        db_transaction.sqlmodel_update(transaction_data)
        session.add(db_transaction)
        session.commit()
        session.refresh(db_transaction)
        return {"status": 200, "data": db_transaction}
     


@app.get("/transactions/")
def read_transactions(
    session: SessionDep,
    offset: int = 0,
    limit: Optional[int] = Query(None, gt=0, le=100), # Optional, no default, between 1 and 100 if provided
) -> dict:
    query = select(Transaction).offset(offset)
    if limit is not None:
        query = query.limit(limit)
    transactions = session.exec(query).all()
    return {"status": 200, "data": transactions}


@app.get("/transactions/{transaction_id}")
def read_transaction(transaction_id: UUID, session: SessionDep) -> dict:
    transaction = session.get(Transaction, transaction_id)
    if not transaction:
        raise HTTPException(status_code=404, detail="Hero not found")
    return {"status": 200, "data": transaction}


@app.delete("/transactions/{transaction_id}")
def delete_transaction(transaction_id: UUID, session: SessionDep) -> dict:
    transaction = session.get(Transaction, transaction_id)
    if not transaction:
        raise HTTPException(status_code=404, detail="Hero not found")
    session.delete(transaction)
    session.commit()
    return {"ok": True}