from fastapi import HTTPException
from sqlalchemy.orm import Session
from models import Account,AccountCustomer,OwnerRole
from schemas import AccountCreate,AccountUpdate

def create_account(db:Session,account:AccountCreate)->Account:
    new_account=Account(
        account_number=account.account_number,
        account_type=account.account_type,
        customer_id=account.customer_id,
        )
    db.add(new_account)
    db.commit()
    db.refresh(new_account)
    return new_account

def get_account_by_id(db:Session,account_id:int)->Account:
    account=db.query(Account).filter(Account.id==account_id).first()
    if account is None:
        raise HTTPException(status_code=404,detail=f"Account with id '{account_id}' not found")
    return account

def get_accounts_for_customer(db:Session,customer_id:int)->list[Account]:
    return db.query(Account).filter(Account.customer_id==customer_id).all()

def get_accounts(db:Session,skip:int=0,limit:int=100)->list[Account]:
    return db.query(Account).offset(skip).limit(limit).all()

def update_account(db:Session,account_id:int,updates:AccountUpdate)->Account:
    existing_account=get_account_by_id(db,account_id)
    updated_data=updates.model_dump(exclude_unset=True)

    for field,value in updated_data.items():
        setattr(existing_account,field,value)
    db.commit()
    db.refresh(existing_account)
    return existing_account

def delete_account(db:Session,account_id:int)->dict:
    account=get_account_by_id(db,account_id)
    account_number=account.account_number
    db.delete(account)
    db.commit()
    return {"message":f"Account {account_number} and its transaction history were deleted"}