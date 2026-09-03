from decimal import Decimal
from sqlalchemy.orm import Session
from fastapi import HTTPException
from models import Transaction,TransactionType,Account
from schemas import DepositRequest,WithdrawRequest,ReversalRequest

REVERSAL_TYPE_MAP = {
    TransactionType.deposit: TransactionType.reversal_deposit,
    TransactionType.withdrawal: TransactionType.reversal_withdrawal,
    TransactionType.transfer_in: TransactionType.reversal_transfer_in,
    TransactionType.transfer_out: TransactionType.reversal_transfer_out,
 
    # reverse direction — "undo the undo"
    TransactionType.reversal_deposit: TransactionType.deposit,
    TransactionType.reversal_withdrawal: TransactionType.withdrawal,
    TransactionType.reversal_transfer_in: TransactionType.transfer_in,
    TransactionType.reversal_transfer_out: TransactionType.transfer_out,
}

def get_account_balance(db:Session,account_id:int)->Decimal:
    account=db.query(Account).filter(Account.id==account_id).first()
    if account is None:
        raise HTTPException(status_code=404,detail=f"Account {account_id} not found")
    total=sum((t.amount for t in account.transactions),Decimal("0.00"))
    return total

def deposit(db:Session,request:DepositRequest)->Transaction:
    account=db.query(Account).filter(Account.id==request.account_id).first()
    if account is None:
        raise HTTPException(status_code=404,detail=f"Account {request.account_id} not found")
    new_txn=Transaction(
        account_id=request.account_id,
        amount=request.amount,
        type=TransactionType.deposit,
    )
    db.add(new_txn)
    db.commit()
    db.refresh(new_txn)
    return new_txn

def withdraw(db:Session,request:WithdrawRequest)->Transaction:
    account=db.query(Account).filter(Account.id==request.account_id).first()
    if account is None:
        raise HTTPException(status_code=404,detail=f"Account '{request.account_id}' not found")

    current_balance=get_account_balance(db,request.account_id)
    if current_balance<request.amount:
        raise HTTPException(status_code=400,detail="Insufficient funds")
    new_txn=Transaction(
        account_id=request.account_id,
        amount=-request.amount,
        type=TransactionType.withdrawal,
    )
    db.add(new_txn)
    db.commit()
    db.refresh(new_txn)
    return new_txn

def reverse_transaction(db:Session,request:ReversalRequest)->Transaction:
    original=db.query(Transaction).filter(Transaction.id==request.transaction_id).first()
    if original is None:
        raise HTTPException(status_code=404,detail=f"Transaction {request.transaction_id} not found")

    if original.reversal_entries:
        raise HTTPException(status_code=400,detail="This transaction has already been reversed")

    reversal_type=REVERSAL_TYPE_MAP.get(original.type)

    if reversal_type is None:
        raise HTTPException(status_code=400,detail=f"Transaction type '{original.type}' cannot be reversed")

    reversal_txn=Transaction(
        account_id=original.account_id,
        amount=-original.amount,
        type=reversal_type,
        reversed_transaction_id=original.id,
    )
    db.add(reversal_txn)
    db.commit()
    db.refresh(reversal_txn)
    return reversal_txn

