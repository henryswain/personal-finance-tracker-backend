import base64
import json
from uuid import UUID
from models import Transaction
from datetime import datetime
from typing import Annotated, Optional
from fastapi import Depends, HTTPException, Query, APIRouter
from sqlmodel import Session, select, func
from db_context import engine

def get_session():
    with Session(engine) as session:
        yield session

SessionDep = Annotated[Session, Depends(get_session)]

transactions_router = APIRouter()

def encode_cursor(value):
    raw = json.dumps({"date": value})
    return base64.urlsafe_b64encode(raw.encode()).decode()

def decode_cursor(cursor):
    try:
        # Add padding if needed (base64 strings must be multiples of 4)
        padding = 4 - (len(cursor) % 4)
        if padding and padding != 4:
            cursor += '=' * padding        
        raw = base64.urlsafe_b64decode(cursor.encode()).decode()
        payload = json.loads(raw)
        return payload.get("date")
    except Exception as e:
        print(f"Cursor decode error: {e}")
        raise HTTPException(status_code=400, detail="Invalid cursor")

@transactions_router.post("/new")
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

def addConditionsToBaseQuery (prev_cursor, next_cursor, direction, limit, filter_type, filter_value, prev_cursor_id, base_query):
    if filter_type and filter_value:
        base_query = base_query.where(getattr(Transaction, filter_type) == filter_value)
    if direction == "prev":
        base_query = base_query.where(Transaction.date > decode_cursor(prev_cursor)).order_by(Transaction.date.asc()).limit(limit+1)
    elif direction == "next":
        base_query = base_query.where(Transaction.date < decode_cursor(next_cursor)).order_by(Transaction.date.desc()).limit(limit+1)
    elif direction == "refresh":
        base_query = base_query.where(Transaction.date <= decode_cursor(prev_cursor)).order_by(Transaction.date.desc()).limit(limit+1)
    else:
        base_query = base_query.where(Transaction.date < decode_cursor(prev_cursor_id)).order_by(Transaction.date.desc()).limit(limit+1)  
    return base_query 

def getCount (prev_cursor, next_cursor, direction, limit, filter_type, filter_value, prev_cursor_id, base_query):
    if filter_type and filter_value:
        base_query = base_query.where(getattr(Transaction, filter_type) == filter_value)
    if direction == "prev":
        base_query = base_query.where(Transaction.date > decode_cursor(prev_cursor))
    elif direction == "next":
        base_query = base_query.where(Transaction.date >= decode_cursor(next_cursor))
    elif direction == "refresh":
        base_query = base_query.where(Transaction.date > decode_cursor(prev_cursor))
    else:
        base_query = base_query.where(Transaction.date < decode_cursor(prev_cursor_id))
    return base_query 

def performQuery(base_query, direction, session):
    results = session.exec(base_query).all()
    if direction == "prev":
        results = list(reversed(results))
    return results

def update_start_and_end_viewing_numbers(direction, hasmore, count_results, limit, results):
    viewing_start_number = 0
    viewing_end_number = -1
    if direction == "prev" and hasmore:
        viewing_start_number = count_results[0] + 1 - limit
        viewing_end_number = count_results[0]
    if direction == "prev" and not hasmore:
        viewing_end_number = count_results[0]
    if direction == "next" and hasmore:
        viewing_start_number = count_results[0] + 1
        viewing_end_number = count_results[0] + limit
    if direction == "next" and not hasmore:
        viewing_start_number = count_results[0] + 1
        viewing_end_number = count_results[0] + len(results)
    if direction == "refresh" and hasmore:
        viewing_start_number = count_results[0] + 1
        viewing_end_number = count_results[0] + limit
    if direction == "refresh" and not hasmore:
        viewing_start_number = count_results[0] + 1
        viewing_end_number = count_results[0] + len(results)
    if direction != "next" and direction != "prev" and direction != "refresh" and hasmore:
        viewing_end_number = limit
    if direction != "next" and direction != "prev" and direction != "refresh" and not hasmore:
        viewing_end_number = len(results)
    return viewing_start_number, viewing_end_number

@transactions_router.get("/")
def read_transactions(
    session: SessionDep,
    prev_cursor: Optional[str] = None,
    next_cursor: Optional[str] = None,
    direction: Optional[str] = None,
    limit: Optional[int] = Query(None, gt=0, le=100), # Optional, no default, between 1 and 100 if provided,
    filter_type: Optional[str] = None,
    filter_value: Optional[str] = None,
) -> dict:
    # get date and time right now as prev_cursor if no cursor is provided
    prev_cursor_id = encode_cursor(datetime.now().strftime("%Y-%m-%dT%H:%M:%S")) 
    # Calculate running total ordered by date (oldest first)
    running_total = func.sum(Transaction.amount).over(order_by=Transaction.date.asc())


    base_query = select(Transaction, running_total.label("running_total"))
    base_query = addConditionsToBaseQuery(prev_cursor, next_cursor, direction, limit, filter_type, filter_value, prev_cursor_id, base_query)
    # Execute query
    results = performQuery(base_query, direction, session)
    count_base_query = select(func.count(Transaction.id))
    count_base_query = getCount(prev_cursor, next_cursor, direction, limit, filter_type, filter_value, prev_cursor_id, count_base_query)
    count_results = performQuery(count_base_query, direction, session)
    hasmore = len(results) > limit


    viewing_start_number, viewing_end_number = update_start_and_end_viewing_numbers(direction, hasmore, count_results, limit, results)


    next_cursor = None
    prev_cursor = None
    next_cursor_not_encoded = None
    prev_cursor_not_encoded = None

    # Trim the overflow item used for "hasmore" detection
    if hasmore:
        if direction == "prev":
            # After reversal, the overflow item is at the beginning
            results = results[1:]
        else:
            # For "next" or initial load, the overflow item is at the end
            results = results[:-1]

    # if there are more transactions, set next_cursor to end or results - the 1 extra returned 
    # and prev_cursor to the first element

    # this allows users to go forward or backwords using the next_cursor or prev_cursor respectively
    if hasmore:
        print("hasmore: ", hasmore)
        if direction == "prev" or direction == "next" or direction == "refresh":
            next_cursor = encode_cursor(results[-1][0].date)
            next_cursor_not_encoded = results[-1][0].date
            prev_cursor = encode_cursor(results[0][0].date)
            prev_cursor_not_encoded = results[0][0].date
        else: 
            prev_cursor = None
            prev_cursor_not_encoded = None
            next_cursor = encode_cursor(results[-1][0].date)
            next_cursor_not_encoded = results[-1][0].date
    else:
        print("hasmore: ", hasmore)
        if direction == "prev":
            print("results: ", results)
            if len(results) < limit:
                base_query = select(Transaction, running_total.label("running_total"))
                base_query = addConditionsToBaseQuery(None, None, None, limit, filter_type, filter_value, prev_cursor_id, base_query)
                results = performQuery(base_query, None, session)
            prev_cursor = None
            prev_cursor_not_encoded = None
            next_cursor = encode_cursor(results[-1][0].date)
            next_cursor_not_encoded = results[-1][0].date
        elif direction == "next" or direction == "refresh":
            next_cursor = None
            next_cursor_not_encoded = None
            prev_cursor = encode_cursor(results[0][0].date)
            prev_cursor_not_encoded = results[0][0].date
        else:
            prev_cursor = None
            prev_cursor_not_encoded = None
            next_cursor = None
            next_cursor_not_encoded = None

    
    # Format the response
    transactions_with_totals = [
        {
            **transaction.model_dump(),
            "running_total": running_total
        }
        for transaction, running_total in results
    ]
    
    return {"status": 200, 
            "data": transactions_with_totals[:limit], 
            "next": next_cursor, 
            "prev": prev_cursor, 
            "prev_not_encoded": prev_cursor_not_encoded, 
            "next_not_encoded": next_cursor_not_encoded, 
            "start_number": viewing_start_number,
            "end_number": viewing_end_number}


@transactions_router.get("/{transaction_id}")
def read_transaction(transaction_id: UUID, session: SessionDep) -> dict:
    transaction = session.get(Transaction, transaction_id)
    if not transaction:
        raise HTTPException(status_code=404, detail="Hero not found")
    return {"status": 200, "data": transaction}

@transactions_router.patch("/{transaction_id}")
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

@transactions_router.delete("/{transaction_id}")
def delete_transaction(transaction_id: UUID, session: SessionDep) -> dict:
    transaction = session.get(Transaction, transaction_id)
    if not transaction:
        raise HTTPException(status_code=404, detail="Hero not found")
    session.delete(transaction)
    session.commit()
    return {"ok": True}