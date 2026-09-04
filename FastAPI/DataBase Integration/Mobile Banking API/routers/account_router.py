from fastapi import APIRouter,Depends,HTTPException,status
from sqlalchemy.orm import Session
from typing import Annotated

from db.database import get_db
from services import create_customer

database_dependency=Annotated[Session,Depends(get_db)]
