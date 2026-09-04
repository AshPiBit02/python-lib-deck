from fastapi import Header,Depends,HTTPException
from .config import settings
from db.database import get_db
from sqlalchemy.orm import Session
from typing import Annotated

def key_validation(key:str=Header(...))->None:
    if key!=settings.secret_key:
        raise HTTPException(status_code=403,detail="Invalid secret key")

def pin_validation(pin:str=Header(...))->None:
    if pin!=settings.pin:
        raise HTTPException(status_code=403,detail="Incorrect PIN")

database_dependency=Annotated[Session,Depends(get_db)]