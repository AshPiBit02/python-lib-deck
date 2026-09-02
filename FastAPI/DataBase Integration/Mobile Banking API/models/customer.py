from sqlalchemy import (
    Column, Integer, String,Date,func,
    )
from sqlalchemy.orm import relationship
from db.database import Base

class Customer(Base):
    __tablename__="customers"
    id=Column(Integer,primary_key=True,index=True)
    full_name=Column(String(150),nullable=False)
    email=Column(String(200),unique=True,nullable=False)
    phone=Column(String(15),nullable=True)
    created_at=Column(Date,server_default=func.now())

    accounts=relationship("Account",back_populates="primary_owner")

    joint_accounts=relationship("Account",secondary="account_customers",back_populates="joint_owners",viewonly=True)

    audit_logs=relationship("AuditLog",back_populates="customer")





    