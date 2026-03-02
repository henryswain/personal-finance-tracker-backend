
from typing import Annotated, Optional
from models import Pot, User
from uuid import UUID
from jwt_auth import get_current_user
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
def create_pot(
        pot: Pot, 
        current_user: Annotated[User, Depends(get_current_user)],
        session: SessionDep) -> dict:
    now = datetime.now()
    print("now: ", now)
    currentTime = now.strftime("%H:%M:%S")
    print("currentTime: ", currentTime)
    
    # Create a Pot object, not a dictionary
    newPot = Pot(
        username=current_user.username,
        target_amount=pot.target_amount,
        title=pot.title
    )
    
    session.add(newPot)
    session.commit()
    session.refresh(newPot)
    return {"status": 200, "data": newPot}

@pots_router.get("/")
def read_pots(
    current_user: Annotated[User, Depends(get_current_user)],
    session: SessionDep,
    offset: int = 0,
    limit: Optional[int] = Query(None, gt=0, le=100), # Optional, no default, between 1 and 100 if provided,
) -> dict:
    query = select(Pot).where(Pot.username == current_user.username)

    # Apply offset and limit
    if offset:
        query = query.offset(offset)
    if limit is not None:
        query = query.limit(limit)

    results = session.exec(query).all()
    
    return {"status": 200, "data": results}

@pots_router.patch("/{pot_id}")
def update_pot(
        current_user: Annotated[User, Depends(get_current_user)],
        pot_id: UUID, 
        pot: Pot, 
        session: SessionDep) -> dict:
    db_pot = session.exec(select(Pot).where((Pot.pot_id == pot_id) & (Pot.username == current_user.username)))
    if not db_pot:
        raise HTTPException(status_code=404, detail="Pot not found")
    # Get only the fields that were provided in the request
    pot_data = pot.model_dump(exclude_unset=True)
    
    # Update the existing transaction
    db_pot.sqlmodel_update(pot_data)
    session.add(db_pot)
    session.commit()
    session.refresh(db_pot)
    return {"status": 200, "data": db_pot}


@pots_router.delete("/{pot_id}")
def delete_pot(
        pot_id: UUID, 
        current_user: Annotated[User, Depends(get_current_user)],
        session: SessionDep) -> dict:
    pot = session.exec(select(Pot).where((Pot.pot_id == pot_id) & (Pot.username == current_user.username)))
    if not pot:
        raise HTTPException(status_code=404, detail="Pot not found")
    session.delete(pot)
    session.commit()
    return {"ok": True}

@pots_router.get("/{pot_id}")
def get_pot(
        pot_id: UUID, 
        current_user: Annotated[User, Depends(get_current_user)],
        session: SessionDep) -> dict:
    pot = session.exec(select(Pot).where((Pot.pot_id == pot_id) & (Pot.username == current_user.username)))
    return {"data": pot}
    