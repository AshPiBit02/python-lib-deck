from fastapi import APIRouter, Depends
import services
import schemas
from core.dependencies import database_dependency, key_validation, pin_validation

secure_transaction_router = APIRouter(prefix="/transaction", dependencies=[Depends(key_validation)])
pin_secure_transaction_router=APIRouter(prefix="/transaction",dependencies=[Depends(key_validation),Depends(pin_validation)])


@secure_transaction_router.get("/view/balance/{account_id}")
def get_balance(db: database_dependency, account_id: int):
    balance = services.get_account_balance(db, account_id)
    return {"account_id": account_id, "balance": balance}


@pin_secure_transaction_router.post("/deposit", response_model=schemas.TransactionResponse)
def make_deposit(db: database_dependency, request: schemas.DepositRequest):
    return services.deposit(db, request)


@pin_secure_transaction_router.post("/withdraw", response_model=schemas.TransactionResponse)
def make_withdrawal(db: database_dependency, request: schemas.WithdrawRequest):
    return services.withdraw(db, request)


@pin_secure_transaction_router.post("/transfer")
def make_transfer(db: database_dependency, request: schemas.TransferRequest):
    return services.transfer(db, request)


@pin_secure_transaction_router.post("/reverse", response_model=schemas.TransactionResponse)
def reverse(db: database_dependency, request: schemas.ReversalRequest):
    return services.reverse_transaction(db, request)