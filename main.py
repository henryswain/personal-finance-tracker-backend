from typing import Annotated, Optional
from fastapi import Depends, FastAPI, HTTPException, Query
from sqlmodel import Field, Session, SQLModel, create_engine, select, func
import os
from uuid import UUID, uuid4
from pydantic import BaseModel
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi.middleware.cors import CORSMiddleware


load_dotenv()
URL = os.getenv('URL_DATABASE')

engine = create_engine(URL)

connect_args = {"check_same_thread": False}
engine = create_engine(URL)

class Transaction(SQLModel, table=True):
    id: UUID | None = Field(default_factory=uuid4, primary_key=True)
    amount: int | None = Field(index=True)
    title: str | None = Field(index = True)
    memo: str | None = Field(default=None, index=True)
    account_name: str | None = Field(default=None, index=True)
    category: str | None = Field(default=None, index=True)
    date: str | None = Field(default=None, index=True)

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


@app.post("/transaction")
def create_transaction(transaction: Transaction, session: SessionDep) -> dict:
    now = datetime.now()
    print("now: ", now)
    currentTime = now.strftime("%H:%M:%S")
    print("currentTime: ", currentTime)
    
    # Create a Transaction object, not a dictionary
    newTransaction = Transaction(
        amount=transaction.amount,
        title=transaction.title,
        memo=transaction.memo,
        account_name=transaction.account_name,
        category=transaction.category,
        date=transaction.date + "T" + currentTime
    )
    
    session.add(newTransaction)
    session.commit()
    session.refresh(newTransaction)
    return {"status": 200, "data": newTransaction}

@app.patch("/transactions/{transaction_id}")
def update_transaction(transaction_id: UUID, transaction: Transaction, session: SessionDep) -> dict:
    db_transaction = session.get(Transaction, transaction_id)
    if not db_transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    now = datetime.now()
    print("now: ", now)
    currentTime = now.strftime("%H:%M:%S")
    print("currentTime: ", currentTime)
    
    # Get only the fields that were provided in the request
    transaction_data = transaction.model_dump(exclude_unset=True)
    
    # If date is being updated, append current time to it
    transaction_data["date"] = transaction_data["date"] + "T" + currentTime
    
    # Update the existing transaction
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
    
    # Calculate running total ordered by date (oldest first)
    running_total = func.sum(Transaction.amount).over(order_by=Transaction.date.asc())
    
    # Select transaction with running total, ordered by date ascending for calculation
    query = select(Transaction, running_total.label('running_total')).order_by(Transaction.date.asc())
    
    # Apply offset and limit
    if offset:
        query = query.offset(offset)
    if limit is not None:
        query = query.limit(limit)
    
    # Execute query
    results = session.exec(query).all()
    
    # Reverse to show newest first, but running total is still calculated from oldest
    results_reversed = list(reversed(results))
    
    # Format the response
    transactions_with_totals = [
        {
            **transaction.model_dump(),
            "running_total": running_total
        }
        for transaction, running_total in results_reversed
    ]
    
    return {"status": 200, "data": transactions_with_totals}


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