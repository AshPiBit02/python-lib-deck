from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy import func
from models import Customer
from schemas import CustomerCreate,CustomerUpdate
from fastapi import HTTPException


def create_customer(db:Session,customer:CustomerCreate)->Customer:
    new_customer=Customer(full_name=customer.full_name,email=customer.email,phone=customer.phone)
    db.add(new_customer)
    db.commit()
    db.refresh(new_customer)
    return new_customer

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
    for field,value in updated_data.items():
        setattr(existing_customer,field,value)
    db.commit()
    db.refresh(existing_customer)
    return existing_customer

def get_customer_with_accounts(db:Session,customer_id:int)->Customer:
    customer=get_customer_by_id(db,customer_id)
    _=customer.accounts
    return customer