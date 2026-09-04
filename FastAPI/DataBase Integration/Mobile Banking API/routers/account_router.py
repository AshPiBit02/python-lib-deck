from fastapi import APIRouter,Header,HTTPException
from typing import Annotated
from fastapi import Depends
from sqlalchemy.orm import Session
import services
import schemas
from core.dependencies import database_dependency,key_validation

account_router=APIRouter(prefix="/account")
secure_account_router=APIRouter(prefix="/account",dependencies=[Depends(key_validation)])

@account_router.get("/view/list",response_model=list[schemas.AccountResponse])
def list_accounts(db:database_dependency,skip:int=0,limit:int=100):
    return services.get_accounts(db,skip,limit)

@account_router.get("/view/{account_id}",response_model=schemas.AccountResponse)
def get_account(db:database_dependency,account_id:int):
    return services.get_account_by_id(db,account_id)

@account_router.get("/view/{customer_id}",response_model=schemas.AccountResponse)
def get_customer_account(db:database_dependency,customer_id:int):
    return services.get_accounts_for_customer(db,customer_id)

@secure_account_router.post("/add",response_model=schemas.AccountResponse)
def add_account(db:database_dependency,account:schemas.AccountCreate):
    return services.create_account(db,account)

@secure_account_router.patch("/udpate",response_model=schemas.AccountResponse)
def update_account(db:database_dependency,account_id:int,updates:schemas.AccountResponse):
    return services.update_account(db,account_id,updates)