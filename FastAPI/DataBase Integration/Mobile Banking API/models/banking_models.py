from sqlalchemy import (
    Column, Integer, String,DateTime,func,ForeignKey,Numeric,Index,
    Boolean,Text
    )
from sqlalchemy.orm import relationship
from sqlalchemy.types import Enum
from db.database import Base
from enum import Enum as PyEnum

class AccountType(str,PyEnum):
    checking="Checking"
    savings="Savings"

class TransactionType(str,PyEnum):
    deposit="Deposit"
    withdrawal="Withdrawal"
    transfer_in="Transfer_In"
    transfer_out="Transfer_Out"

class OwnerRole(str,PyEnum):
    primary="Primary"
    joint="Joint"

class LogStatus(str,PyEnum):
    success="Success"
    failed="Failed"

class Customer(Base):
    __tablename__="customers"
    id=Column(Integer,primary_key=True,index=True)
    full_name=Column(String(150),nullable=False)
    email=Column(String(200),unique=True,nullable=False)
    phone=Column(String(15),nullable=True)
    created_at=Column(DateTime,server_default=func.now())

    # One-to-Many: Customer->Account(as primary owner)
    accounts=relationship("Account",back_populates="primary_owner")

    # Many-to-Many: Customer<->Account(joint ownership, via association table)
    joint_accounts=relationship("Account",secondary="account_customers",back_populates="joint_owners",viewonly=True)

    # One-to-Many: Customer->AuditLog
    audit_logs=relationship("AuditLog",back_populates="customer")


class Account(Base):
    __tablename__="accounts"
    id=Column(Integer,primary_key=True,index=True)
    account_number=Column(String(20),unique=True,nullable=False)
    account_type=Column(Enum(AccountType),nullable=False)
    customer_id=Column(Integer,ForeignKey("customers.id"),nullable=False)
    created_at=Column(DateTime,server_default=func.now())

    # Many-to-One: Account-> Customer(primary owner)
    primary_owner=relationship("Customer",back_populates="accounts")

    # One-to-Many: Account -> Transaction
    transactions=relationship("Transaction",back_populates="account",cascade="all,delete-orphan")

    # One-to-One: Account -> Card
    card=relationship("Card",back_populates="account",uselist=False)

    # Many-to-Many: Account <-> Customer (joint owners), via association table
    joint_owners=relationship("Customer",secondary="account_customers",back_populates="joint_accounts",viewonly=True)


class AccountCustomer(Base):
    __tablename__="account_customers"
    account_id=Column(Integer,ForeignKey("accounts.id"),primary_key=True)
    customer_id=Column(Integer,ForeignKey("customers.id"),primary_key=True)
    role=Column(Enum(OwnerRole),nullable=False,default=OwnerRole.joint)

class Transaction(Base):
    __tablename__="transactions"
    id=Column(Integer,primary_key=True,index=True)
    account_id=Column(Integer,ForeignKey("accounts.id"),nullable=False)
    amount=Column(Numeric(12,2),nullable=False)
    type=Column(Enum(TransactionType),nullable=False)

    # self-referencing FK
    reversed_transaction_id=Column(Integer,ForeignKey("transactions.id"),nullable=True)

    created_at=Column(DateTime,server_default=func.now())

    # Many-to-One: Transaction -> Account
    account=relationship("Account",back_populates="transactions")

    # self-referencing relationship: this transaction's reversal target,
    # and (reverse direction) any transaction that reversed THIS one
    reversed_transaction=relationship("Transaction",remote_side=[id],backref="reversed_entries")

    __table_args__=(Index("ix_txn_account_created","account_id","created_at"),)


class Card(Base):
    __tablename__="cards"
    id=Column(Integer,primary_key=True,index=True)
    account_id=Column(Integer,ForeignKey("accounts.id"),unique=True,nullable=False)
    card_number=Column(String(20),unique=True,nullable=False)
    expiry_date=Column(String(7),nullable=False)
    is_active=Column(Boolean,default=True)
    account=relationship("Account",back_populates="card")


class AuditLog(Base):
    __tablename__="audit_logs"
    id=Column(Integer,primary_key=True,index=False)
    action=Column(String(100),nullable=False)
    customer_id=Column(Integer,ForeignKey("customers.id"),nullable=True) # nullable - system actions
    details=Column(Text,name=True)
    status=Column(Enum(LogStatus),nullable=False,default=LogStatus.success)
    created_at=Column(DateTime,server_default=func.now())
    customer=relationship("Customer",back_populates="audit_logs")