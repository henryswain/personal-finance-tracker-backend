
from models import PotItem, Pot
from datetime import datetime
from uuid import UUID, uuid4
from sqlalchemy import case
from typing import Annotated, Optional
from fastapi import Depends, HTTPException, Query, APIRouter
from sqlmodel import Session, select, func
from db_context import engine

def get_session():
    with Session(engine) as session:
        yield session

SessionDep = Annotated[Session, Depends(get_session)]

pot_items_router = APIRouter()

@pot_items_router.post("/new/{pot_id}")
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


@pot_items_router.get("/pots/{pot_id}")
def read_pot_items(
    pot_id: str,
    session: SessionDep,
    offset: int = 0,
    limit: Optional[int] = Query(None, gt=0, le=100), # Optional, no default, between 1 and 100 if provided,
) -> dict:
    # Get the pot to verify it exists and get its title (which is the account name)
    pot = session.get(Pot, pot_id)
    if not pot:
        raise HTTPException(status_code=404, detail="Pot not found")
    
    # Use the pot's title as the account name to match against transfer_to/transfer_from
    account_name = pot.title
    print("account_name: ", account_name)
    
    # Calculate signed amount: positive if transfer_to matches account_name, negative if transfer_from matches
    signed_amount = case(
        (PotItem.transfer_to == account_name, PotItem.amount),
        (PotItem.transfer_from == account_name, -PotItem.amount),
        else_=0
    )
    
    # Calculate running total ordered by date (oldest first)
    running_total = func.sum(signed_amount).over(order_by=PotItem.date.asc())
    
    # Filter to only items where account_name is involved in transfer_to OR transfer_from
    query = select(PotItem, running_total.label('running_total')).where(
        (PotItem.transfer_to == account_name) | (PotItem.transfer_from == account_name)
    ).order_by(PotItem.date.asc())

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
    pot_items_with_totals = [
        {
            **pot_item.model_dump(),
            "running_total": running_total
        }
        for pot_item, running_total in results_reversed
    ]
    
    return {"status": 200, "data": pot_items_with_totals}

@pot_items_router.get("/{pot_item_id}")
def read_pot_item(pot_item_id: UUID, session: SessionDep) -> dict:
    pot_item = session.get(PotItem, pot_item_id)
    if not pot_item:
        raise HTTPException(status_code=404, detail="Hero not found")
    return {"status": 200, "data": pot_item}

@pot_items_router.patch("/{pot_item_id}")
def update_pot_item(pot_item_id: UUID, pot_item: PotItem, session: SessionDep) -> dict:
    # Get the pot item using pot_item_id from URL
    db_pot_item = session.get(PotItem, pot_item_id)
    if not db_pot_item:
        raise HTTPException(status_code=404, detail="Pot item not found")
    
    # Store the old amount before updating
    previous_amount = db_pot_item.amount
    
    # Get the pot
    db_pot = session.get(Pot, db_pot_item.pot_id)
    if not db_pot:
        raise HTTPException(status_code=404, detail="Pot not found")
    
    db_pot.amount = db_pot.amount - previous_amount + pot_item.amount
    
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

@pot_items_router.delete("/{pot_item_id}")
def delete_pot_item(pot_item_id: UUID, session: SessionDep) -> dict:
    pot_item = session.get(PotItem, pot_item_id)
    if not pot_item:
        raise HTTPException(status_code=404, detail="Pot item not found")
    session.delete(pot_item)
    session.commit()
    return {"ok": True}