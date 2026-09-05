from fastapi import HTTPException
from sqlalchemy.orm import Session
from models import Account,AccountCustomer,OwnerRole
from schemas import AccountCreate,AccountUpdate,JointOwnerAdd
from services.audit_service import log_action,LogStatus

def create_account(db:Session,account:AccountCreate)->Account:
    new_account=Account(
        account_number=account.account_number,
        account_type=account.account_type,
        customer_id=account.customer_id,
        )
    try:
        db.add(new_account)
        log_action(db,"account_creation",account.customer_id,f"Account created: {account.account_number}",LogStatus.success)
        db.commit()
        db.refresh(new_account)
        return new_account
    except Exception as e:
        db.rollback()
        log_action(db,"account_creation",account.customer_id,f"Faild to create account",LogStatus.failed,commit_independently=True)
        raise HTTPException(status_code=400,detail="Failed to create account")

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
    try:
        for field,value in updated_data.items():
            setattr(existing_account,field,value)
        log_action(db,"account_update",existing_account.customer_id,f"Fields updated: {list(updated_data.keys())}",LogStatus.success)
        db.commit()
        db.refresh(existing_account)
        return existing_account
    except Exception as e:
        db.rollback()
        log_action(db,"account_update",existing_account.customer_id,f"Failed to update account {account_id}: {str(e)}",LogStatus.success,commit_independently=True)
        raise HTTPException(status_code=400,detail="Failed to update customer")
        

def delete_account(db:Session,account_id:int)->dict:
    account=get_account_by_id(db,account_id)
    account_number=account.account_number
    try:
        db.delete(account)
        log_action(db,"account_deletion",account.customer_id,f"Delete account: {account.id}",LogStatus.success)
        db.commit()
        return {"message":f"Account {account_number} and its transaction history were deleted"}
    except Exception as e:
        db.rollback()
        log_action(db,"account_deletion",account.customer_id,f"Failed to delete account {account.id}: {str(e)}",LogStatus.failed,commit_independently=True)
        raise HTTPException(status_code=400,detail="Failed to delete account")
    
def add_joint_owner(db:Session,account_id:int,request:JointOwnerAdd)->AccountCustomer:
    account=db.query(Account).filter(Account.id==account_id).first()
    if account is None:
        log_action(db,"joint_owner_addition",None,f"Failed: account {account_id} not found",LogStatus.failed,commit_independently=True)
        raise HTTPException(status_code=404,detail=f"Account {account_id} not found")

    if account.customer_id==request.customer_id:
        log_action(db,"joint_owner_addition",request.customer_id,f"Failed: Customer is already the primary owner of this account",LogStatus.failed,commit_independently=True)
        raise HTTPException(status_code=400,detail=f"Customer is already the primary owner of this account")

    existing=db.query(AccountCustomer).filter(AccountCustomer.account_id==account_id,
                                              AccountCustomer.customer_id==request.customer_id,).first()

    if existing is not None:
        raise HTTPException(status_code=400,detail="Customer is already a joint owner of this account")

    link=AccountCustomer(
        account_id=account_id,
        customer_id=request.customer_id,
        role=request.role,
    )

    try:
        db.add(link)
        log_action(db,"joint_owner_addtion",request.customer_id,f"Customer {request.customer_id} added as {request.role} owner on account {account_id}",LogStatus.success)
        db.commit()
        db.refresh(link)
        return link
    except Exception as e:
        db.rollback()
        log_action(db,"joint_owner_addition",request.customer_id,f"Failed to add joint owner: {str(e)}",LogStatus.failed,commit_independently=True)
        raise HTTPException(status_code=400,detail="Failed to add joint owner")

def get_joint_owners(db:Session,account_id:int)->list[AccountCustomer]:
    return db.query(AccountCustomer).filter(AccountCustomer.id==account_id).all()

def remove_joint_owner(db:Session,account_id:int,customer_id:int)->dict:
    link=db.query(AccountCustomer).filter(AccountCustomer.id==account_id,AccountCustomer.customer_id==customer_id,).first()
    if link is None:
        raise HTTPException(status_code=404,detail="This customer is not a joint owner of this account")
    try:
        db.delete(link)
        log_action(db,"joint_owner_remove",customer_id,f"Customer {customer_id} removed as joint owner on account {account_id}",LogStatus.success)
        db.commit()
        return {"message":f"Customer {customer_id} removed as joint owner on account {account_id}"}
    except Exception as e:
        db.rollback()
        log_action(db,"joint_owner_remove",customer_id,f"Failed to remove joint owner: {str(e)}",LogStatus.failed,commit_independently=True)
        raise HTTPException(status_code=400,detail="Failed to remove joint owner")