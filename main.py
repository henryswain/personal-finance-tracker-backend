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


class Pot(SQLModel, table=True):
    pot_id: UUID | None = Field(default_factory=uuid4, primary_key=True)
    title: str | None = Field(index=True)
    target_amount: int | None = Field(index=True)
    amount: int | None = Field(default_factory=lambda: 0, index=True)


class PotItem(SQLModel, table=True):
    pot_item_id: UUID | None = Field(default_factory=uuid4, primary_key=True)
    pot_id: UUID = Field(foreign_key="pot.pot_id", index=True)
    amount: int | None = Field(index=True)
    title: str | None = Field(index=True)
    transfer_from: str | None = Field(index=True)
    transfer_to: str | None = Field(index=True)
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

@app.post("/pot")
def create_pot(pot: Pot, session: SessionDep) -> dict:
    now = datetime.now()
    print("now: ", now)
    currentTime = now.strftime("%H:%M:%S")
    print("currentTime: ", currentTime)
    
    # Create a Pot object, not a dictionary
    newPot = Pot(
        target_amount=pot.target_amount,
        title=pot.title
    )
    
    session.add(newPot)
    session.commit()
    session.refresh(newPot)
    return {"status": 200, "data": newPot}

@app.delete("/pots/{pot_id}")
def delete_pot(pot_id: UUID, session: SessionDep) -> dict:
    pot = session.get(Pot, pot_id)
    if not pot:
        raise HTTPException(status_code=404, detail="Hero not found")
    session.delete(pot)
    session.commit()
    return {"ok": True}

@app.delete("/pot_items/{pot_id}")
def delete_pot_item(pot_item_id: UUID, session: SessionDep) -> dict:
    pot_item = session.get(Pot, pot_item_id)
    if not pot_item:
        raise HTTPException(status_code=404, detail="Hero not found")
    session.delete(pot_item)
    session.commit()
    return {"ok": True}

@app.get("/pots/{pot_id}")
def get_pot(pot_id: UUID, session: SessionDep) -> dict:
    pot = session.get(Pot, pot_id)
    return {"data": pot}

@app.post("/pot_item/{pot_id}")
def create_pot_item(pot_item: PotItem, pot_id: str, session: SessionDep) -> dict:
    db_pot = session.exec(select(Pot).where(Pot.pot_id == pot_id)).first()
    if not db_pot:
        raise HTTPException(status_code=404, detail="Pot not found")
    
    db_pot.amount = db_pot.amount + pot_item.amount
    now = datetime.now()
    print("now: ", now)
    currentTime = now.strftime("%H:%M:%S")
    
    # Create a PotItem object
    newPotItem = PotItem(
        pot_id=pot_id,
        amount=pot_item.amount,
        title=pot_item.title,
        transfer_from=pot_item.transfer_from,
        transfer_to=pot_item.transfer_to,
        date=pot_item.date + "T" + currentTime
    )
    
    session.add(newPotItem)
    session.add(db_pot)
    session.commit()
    session.refresh(newPotItem)
    return {"status": 200, "data": newPotItem}

@app.patch("/pot_items/{pot_id}")
def update_pot_item(pot_id: UUID, pot_item: PotItem, prevous_pot_item_amount: int, session: SessionDep) -> dict:
    print("prevous_pot_item_amount: ", prevous_pot_item_amount)
    print("pot_id: ", pot_id)
    
    # Get the pot directly using pot_id from URL
    db_pot = session.get(Pot, pot_id)
    print("db_pot: ", db_pot)
    if not db_pot:
        raise HTTPException(status_code=404, detail="Pot not found")
    
    # Get the pot item using pot_item_id from the request body
    db_pot_item = session.exec(select(PotItem).where(PotItem.pot_id == pot_id)).first()
    print("db_pot_item: ", db_pot_item)
    if not db_pot_item:
        raise HTTPException(status_code=404, detail="pot item not found")
    
    # Update the pot's total amount
    db_pot.amount = db_pot.amount - prevous_pot_item_amount + pot_item.amount
    
    now = datetime.now()
    print("now: ", now)
    currentTime = now.strftime("%H:%M:%S")
    print("currentTime: ", currentTime)
    
    # Get only the fields that were provided in the request
    pot_item_data = pot_item.model_dump(exclude_unset=True)
    
    if "date" in pot_item_data:
        pot_item_data["date"] = pot_item_data["date"] + "T" + currentTime
    
    # Update the existing pot item
    db_pot_item.sqlmodel_update(pot_item_data)
    session.add(db_pot_item)
    session.add(db_pot)
    session.commit()
    session.refresh(db_pot_item)
    return {"status": 200, "data": db_pot_item}

@app.patch("/pots/{pot_id}")
def update_pot(pot_id: UUID, pot: Pot, session: SessionDep) -> dict:
    db_pot = session.get(Pot, pot_id)
    # Get only the fields that were provided in the request
    pot_data = pot.model_dump(exclude_unset=True)
    
    # Update the existing transaction
    db_pot.sqlmodel_update(pot_data)
    session.add(db_pot)
    session.commit()
    session.refresh(db_pot)
    return {"status": 200, "data": db_pot}

@app.get("/pot_items/{pot_id}")
def read_pot_items(
    pot_id: str,
    session: SessionDep,
    offset: int = 0,
    limit: Optional[int] = Query(None, gt=0, le=100), # Optional, no default, between 1 and 100 if provided,
) -> dict:
    
    # Calculate running total ordered by date (oldest first)
    running_total = func.sum(PotItem.amount).over(order_by=PotItem.date.asc())
    

    query = select(PotItem, running_total.label('running_total')).where(Pot.pot_id == pot_id).order_by(PotItem.date.asc())

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
    pots_with_totals = [
        {
            **pot.model_dump(),
            "running_total": running_total
        }
        for pot, running_total in results_reversed
    ]
    
    return {"status": 200, "data": pots_with_totals}

@app.get("/pots")
def read_pots(
    session: SessionDep,
    offset: int = 0,
    limit: Optional[int] = Query(None, gt=0, le=100), # Optional, no default, between 1 and 100 if provided,
) -> dict:


    query = select(Pot)

    # Apply offset and limit
    if offset:
        query = query.offset(offset)
    if limit is not None:
        query = query.limit(limit)

    results = session.exec(query).all()
    
    return {"status": 200, "data": results}



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
    limit: Optional[int] = Query(None, gt=0, le=100), # Optional, no default, between 1 and 100 if provided,
    filter_type: Optional[str] = None,
    filter_value: Optional[str] = None
) -> dict:
    
    # Calculate running total ordered by date (oldest first)
    running_total = func.sum(Transaction.amount).over(order_by=Transaction.date.asc())
    
    if filter_type and filter_value:
        query = select(Transaction, running_total.label("running_total")).where(getattr(Transaction, filter_type) == filter_value).order_by(Transaction.date.asc())
    else:
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