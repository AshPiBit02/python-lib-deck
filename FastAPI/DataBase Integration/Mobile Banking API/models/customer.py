from sqlalchemy import (
    Column, Integer, String,Date,func,
    )
from sqlalchemy.orm import relationship
from db.database import Base
from enum import Enum

class AccounType(str,Enum):
    checking="Checking"
    savings="Savings"

class TransactionType(str,Enum):
    deposit="Deposit"
    withdrawal="Withdrawal"
    transfer_in="Transfer_In"
    transfer_out="Transfer_Out"

class OwnerRole(str,Enum):
    primary="Primary"
    joint="Joint"

class LogStatus(str,Enum):
    success="Success"
    failed="Failed"

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


class Account(Base):
    __tablename__="accounts"
    id=Column(Integer,primary_key=True,index=True)
    account_numer=Column(String(20),unique=True,nullable=False)
    account_type=Column(SqlEnum(AccountType),nullable=False)



    