from fastapi import FastAPI
from pots import pots_router
from pot_items import pot_items_router
from transactions import transactions_router
from jwt_auth import users_router
from contextlib import asynccontextmanager
from db_context import create_db_and_tables

from fastapi.middleware.cors import CORSMiddleware



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

# Include the movie router
app.include_router(transactions_router, tags=["transactions"], prefix="/transactions")
app.include_router(pots_router, tags=["pots"], prefix="/pots")
app.include_router(pot_items_router, tags=["pot_items"], prefix="/pot_items")
app.include_router(users_router, tags=["users"], prefix="/users")