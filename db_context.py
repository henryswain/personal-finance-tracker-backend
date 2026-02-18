from dotenv import load_dotenv
import os
from sqlmodel import Session, SQLModel, create_engine

def get_session():
    with Session(engine) as session:
        yield session


load_dotenv()
URL = os.getenv('URL_DATABASE')
engine = create_engine(URL)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)