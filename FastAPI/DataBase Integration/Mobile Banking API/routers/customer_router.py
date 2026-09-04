from fastapi import APIRouter,Header,HTTPException
from typing import Annotated
from db.database import get_db
from fastapi import Depends
from sqlalchemy.orm import Session
from core.config import Settings
import services
import schemas

database_dependency=Annotated[Session,Depends(get_db)]

def key_validation(key:str=Header(...)):
    if key!=Settings.secret_key:
        raise HTTPException(status_code=403,detail="Invalid secret key!")

def pin_validation(pin:str=Header(...)):
    if pin!=Settings.pin:
        raise HTTPException(status_code=403,detail="Incorrect PIN")

customer_router=APIRouter(prefix="/customer")
secure_customer_router=APIRouter(prefix="/customer",dependencies=[Depends(key_validation)])
pin_secure_customer_router=APIRouter(prefix="/customer",dependencies=[Depends(key_validation),Depends(pin_validation)])
