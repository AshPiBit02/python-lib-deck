from fastapi import APIRouter,Header,HTTPException
from fastapi import Depends
import services
import schemas
from core.dependencies import database_dependency,key_validation

account_router=APIRouter(prefix="/account")
secure_account_router=APIRouter(prefix="/account",dependencies=[Depends(key_validation)])

@account_router.get("/view/list",response_model=list[schemas.AccountResponse])
def list_accounts(db:database_dependency,skip:int=0,limit:int=100):
    return services.get_accounts(db,skip,limit)

@account_router.get("/view/jointOwners",response_model=list[schemas.JointOwnerResponse])
def get_joint_owners(db:database_dependency,account_id:int):
    return services.get_joint_owners(db,account_id)

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

@secure_account_router.delete("/delete/{account_id}")
def delete_account(db:database_dependency,account_id:int):
    return services.delete_account(db,account_id)


@secure_account_router.post("/add/jointOwner",response_model=schemas.JointOwnerResponse)
def add_joint_owner(db:database_dependency,account_id:int,request:schemas.JointOwnerAdd):
    return services.add_joint_owner(db,account_id,request)


@secure_account_router.delete("/delete/jointOwner")
def delete_joint_owner(db:database_dependency,account_id:int,customer_id:int):
    return services.remove_joint_owner(db,account_id,customer_id)

@secure_account_router.patch("/freeze/{account_id}", response_model=schemas.AccountResponse)
def freeze(db: database_dependency, account_id: int):
    return services.freeze_account(db, account_id)

@secure_account_router.patch("/unfreeze/{account_id}", response_model=schemas.AccountResponse)
def unfreeze(db: database_dependency, account_id: int):
    return services.unfreeze_account(db, account_id)