
from typing import Annotated, Optional
from models import Pot
from uuid import UUID

from fastapi import Depends, HTTPException, Query, APIRouter
from datetime import datetime
from sqlmodel import Session, select
from db_context import engine

def get_session():
    with Session(engine) as session:
        yield session

SessionDep = Annotated[Session, Depends(get_session)]

pots_router = APIRouter()

@pots_router.post("/new")
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

@pots_router.get("/")
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

@pots_router.patch("/{pot_id}")
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


@pots_router.delete("/{pot_id}")
def delete_pot(pot_id: UUID, session: SessionDep) -> dict:
    pot = session.get(Pot, pot_id)
    if not pot:
        raise HTTPException(status_code=404, detail="Hero not found")
    session.delete(pot)
    session.commit()
    return {"ok": True}

@pots_router.get("/{pot_id}")
def get_pot(pot_id: UUID, session: SessionDep) -> dict:
    pot = session.get(Pot, pot_id)
    return {"data": pot}
    