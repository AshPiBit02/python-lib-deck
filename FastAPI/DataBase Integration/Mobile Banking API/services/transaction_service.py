from decimal import Decimal
from sqlalchemy.orm import Session
from fastapi import HTTPException
from models import Transaction,TransactionType,Account
from schemas import DepositRequest,WithdrawRequest,ReversalRequest,TransferRequest
from services.audit_service import log_action,LogStatus

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
        log_action(db,"deposit",None,f"Failed: account {request.account_id} not found",LogStatus.failed,commit_independently=True)
        raise HTTPException(status_code=404,detail=f"Account {request.account_id} not found")
    new_txn=Transaction(
        account_id=request.account_id,
        amount=request.amount,
        type=TransactionType.deposit,
    )
    db.add(new_txn)
    log_action(db,"deposit",account.customer_id,f"Deposited {request.amount} to account {request.account_id}",LogStatus.success)
    db.commit()
    db.refresh(new_txn)
    return new_txn
        

def withdraw(db:Session,request:WithdrawRequest)->Transaction:
    account=db.query(Account).filter(Account.id==request.account_id).first()
    if account is None:
        log_action(db,"withdraw",None,f"Failed: account {request.account_id} not found",LogStatus.failed,commit_independently=True)
        raise HTTPException(status_code=404,detail=f"Account '{request.account_id}' not found")

    current_balance=get_account_balance(db,request.account_id)
    if current_balance<request.amount:
        log_action(db,"withdraw",account.customer_id,f"Failed: insufficient funds in account {request.account_id} (balance ${current_balance}), requested ${request.amount}",LogStatus.failed,commit_independently=True)
        raise HTTPException(status_code=400,detail="Insufficient funds")
    new_txn=Transaction(
        account_id=request.account_id,
        amount=-request.amount,
        type=TransactionType.withdrawal,
    )
    db.add(new_txn)
    log_action(db,"withdraw",account.customer_id,f"Withdrew {request.amount} from account {request.account_id}",LogStatus.success)
    db.commit()
    db.refresh(new_txn)
    return new_txn

def reverse_transaction(db:Session,request:ReversalRequest)->Transaction:
    original=db.query(Transaction).filter(Transaction.id==request.transaction_id).first()
    if original is None:
        log_action(db, "transaction_reversal", None, f"Failed: transaction {request.transaction_id} not found", LogStatus.failed, commit_independently=True)
        raise HTTPException(status_code=404,detail=f"Transaction {request.transaction_id} not found")

    if original.reversed_entries:
        log_action(db, "transaction_reversal", original.account.customer_id, f"Failed: transaction {request.transaction_id} has already been reversed", LogStatus.failed, commit_independently=True)
        raise HTTPException(status_code=400,detail="This transaction has already been reversed")

    reversal_type=REVERSAL_TYPE_MAP.get(original.type)

    if reversal_type is None:
        log_action(db, "transaction_reversal", original.account.customer_id, f"Failed: transaction type '{original.type}' cannot be reversed", LogStatus.failed, commit_independently=True)
        raise HTTPException(status_code=400,detail=f"Transaction type '{original.type}' cannot be reversed")

    reversal_txn=Transaction(
        account_id=original.account_id,
        amount=-original.amount,
        type=reversal_type,
        reversed_transaction_id=original.id,
    )
    db.add(reversal_txn)
    log_action(db, "transaction_reversal", original.account.customer_id, f"Reversed transaction {original.id} (new transaction {reversal_txn.id})", LogStatus.success)
    db.commit()
    db.refresh(reversal_txn)
    return reversal_txn

def transfer(db:Session,request:TransferRequest)->dict:
    from_account=db.query(Account).filter(Account.id==request.from_account_id).first()
    if from_account is None:
        log_action(db, "transfer", None, f"Failed: source account {request.from_account_id} not found", LogStatus.failed, commit_independently=True)
        raise HTTPException(status_code=404,detail=f"Account {request.from_account_id} not found")

    to_account=db.query(Account).filter(Account.id==request.to_account_id).first()
    if to_account is None:
        log_action(db, "transfer", from_account.customer_id, f"Failed: destination account {request.to_account_id} not found", LogStatus.failed, commit_independently=True)
        raise HTTPException(status_code=404,detail=f"Account {request.to_account_id} not found")

    if request.from_account_id==request.to_account_id:
        log_action(db, "transfer", from_account.customer_id, f"Failed: cannot transfer to same account {request.from_account_id}", LogStatus.failed, commit_independently=True)
        raise HTTPException(status_code=400,detail="Cannot transfer to the same account")

    current_balance=get_account_balance(db,request.from_account_id)
    if current_balance<request.amount:
        log_action(db, "transfer", from_account.customer_id, f"Failed: insufficient funds in account {request.from_account_id} (balance {current_balance}, requested {request.amount})", LogStatus.failed, commit_independently=True)
        raise HTTPException(status_code=400,detail="Insufficient funds")

    try:
        debit_txn=Transaction(
            account_id=request.from_account_id,
            amount=-request.amount,
            type=TransactionType.transfer_out,
        )
        credit_txn=Transaction(
            account_id=request.to_account_id,
            amount=request.amount,
            type=TransactionType.transfer_in,
        )
        db.add(debit_txn)
        db.add(credit_txn)
        log_action(db, "transfer", from_account.customer_id, f"Transferred {request.amount} from account {request.from_account_id} to account {request.to_account_id}", LogStatus.success)
        db.commit()
        db.refresh(debit_txn)
        db.refresh(credit_txn)
        return {
            "message":f"Transferred {request.amount} from account {request.from_account_id} to {request.to_account_id}",
            "debit_transaction_id":debit_txn.id,
            "credit_transaction_id":credit_txn.id,
        }
    except Exception:
        db.rollback()
        log_action(db, "transfer", from_account.customer_id, f"Failed: transfer from {request.from_account_id} to {request.to_account_id} could not be completed", LogStatus.failed, commit_independently=True)
        raise HTTPException(status_code=500,detail="Transfer failed, no changes were made")
        