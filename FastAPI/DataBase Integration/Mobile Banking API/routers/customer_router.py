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


@customer_router.get("/view/list",response_model=list[schemas.CustomerResponse])
def list_customers(db:database_dependency,skip:int=0,limit:int=100):
    return services.get_customers(db,skip,limit)

@customer_router.get("/view/{customer_id}",response_model=schemas.CustomerResponse)
def get_customer(db:database_dependency,customer_id:int):
    return services.get_customer_by_id(db,customer_id)

@customer_router.get("/view/{customer_id}/accounts",response_model=schemas.CustomerWithAccounts)
def get_customer_accounts(db:database_dependency,customer_id:int):
    return services.get_customers_with_accounts(db,customer_id)

@secure_customer_router.post("/add",response_model=schemas.CustomerResponse)
def add_customer(db:database_dependency,customer:schemas.CustomerCreate):
    return services.create_customer(db,customer)

@secure_customer_router.patch("/update/{customer_id}",response_model=schemas.CustomerResponse)
def update_customer(db:database_dependency,customer_id:int,updates:schemas.CustomerUpdate):
    return services.update_customer(db,customer_id,updates)