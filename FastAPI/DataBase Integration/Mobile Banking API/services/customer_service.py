from sqlalchemy.orm import Session,joinedload
from models import Customer
from schemas import CustomerCreate,CustomerUpdate
from fastapi import HTTPException
from services.audit_service import log_action,LogStatus


def create_customer(db:Session,customer:CustomerCreate)->Customer:
    new_customer=Customer(full_name=customer.full_name,email=customer.email,phone=customer.phone)
    try:
        db.add(new_customer)
        log_action(db,"customer_creation",None,f"New customer: {customer.email}",LogStatus.success)
        db.commit()
        db.refresh(new_customer)
        return new_customer
    except Exception as e:
        db.rollback()
        log_action(db,"customer_creatoin",None,f"Failed to create customer",LogStatus.failed,commit_independently=True)
        raise HTTPException(status_code=400,detail="Failed to create customer")

def get_customer_by_id(db:Session,customer_id:int)->Customer:
    customer=db.query(Customer).filter(Customer.id==customer_id).first()
    if customer is None:
        raise HTTPException(status_code=404,detail=f"Customer with id {customer_id} not found")
    return customer

def get_customer_by_email(db:Session,email:str)->Customer:
    customer=db.query(Customer).filter(Customer.email==email).first()
    if customer is None:
        raise HTTPException(status_code=404,detail=f"Customer with email '{email}' not found")
    return customer

def get_customers(db:Session,skip:int=0,limit:int=100)->list[Customer]:
    customers=db.query(Customer).offset(skip).limit(limit).all()
    if not customers:
        raise HTTPException(status_code=404,detail="No customer exists")
    return customers

def update_customer(db:Session,customer_id:int,updates:CustomerUpdate)->Customer:
    existing_customer=get_customer_by_id(db,customer_id)
    updated_data=updates.model_dump(exclude_unset=True)
    try:
        for field,value in updated_data.items():
            setattr(existing_customer,field,value)
        log_action(db,"customer_update",customer_id,f"Fields updated: {list(updated_data.keys())}",LogStatus.success)
        db.commit()
        db.refresh(existing_customer)
        return existing_customer
    except Exception as e:
        db.rollback()
        log_action(db,"customer_update",customer_id,f"Failed to update customer {customer_id}: {str(e)}",LogStatus.failed,commit_independently=True)
        raise HTTPException(status_code=400,detail="Failed to update customer")

def get_customers_with_accounts(db:Session,skip:int=0,limit:int=100)->list[Customer]:
    customers=db.query(Customer).options(joinedload(Customer.accounts)).offset(skip).limit(limit).all()
    return customers