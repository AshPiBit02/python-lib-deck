from fastapi import FastAPI,Header,HTTPException,Depends,APIRouter
from typing import Annotated
from pydantic import field_validator
from dummy_data import USER_PRIVATE_CREDENITIALS
app=FastAPI()


current_user_log_status={"userid":None,"logged":False}

def user_auth(userid:str=Header(...),password:str=Header(...))->dict:
    if userid not in USER_PRIVATE_CREDENITIALS:
        raise HTTPException(status_code=404,detail="Unknown user!")
    if USER_PRIVATE_CREDENITIALS[userid]["password"]!=password:
        raise HTTPException(status_code=401,detail="Incorrect password!")
    current_user_log_status["userid"]=userid
    current_user_log_status["logged"]=True
    return {"message":"Login successsful"}
def get_user()->str:
    user=current_user_log_status["userid"]
    if not user:
        raise HTTPException(status_code=403,detail="Login to wallet first!")
    return user

user_dependency=Annotated[str,Depends(get_user)]

def user_logged(user:user_dependency)->bool:
    if not current_user_log_status["logged"]:
         return False
    return True

user_logged_dependency=Annotated[bool,Depends(user_logged)]

def validate_pin(user:user_dependency,pin:str=Header(...)):
    if pin != USER_PRIVATE_CREDENITIALS[user]["pin"]:
        raise HTTPException(status_code=401,detail="Incorrect PIN")
    return {"message":"Transaction successful"}

def inputAmount(amount:float=Header(...))->float:
    if amount<=0.0:
        raise HTTPException(status_code=400,detail="Amount must be non-negative")
    return amount

pin_validation_dependency=Annotated[dict,Depends(validate_pin)]
login_user_dependency=Annotated[dict,Depends(user_auth)]
input_amount_dependency=Annotated[float,Depends(inputAmount)]

@app.get("/wallet/login")
def login(res:login_user_dependency,user:user_dependency):
    return {"message":f"Welcome, Sir({user}). Your current balance is {USER_PRIVATE_CREDENITIALS[user]['balance']}$"}

@app.get("/wallet/transaction/withdraw")
def transfer(user:user_dependency,logged:user_logged_dependency,amount:input_amount_dependency):
    current_balance=USER_PRIVATE_CREDENITIALS[user]["balance"]
    if current_balance<amount:
        raise HTTPException(status_code=400,detail="Insufficient Balance!")
    new_balance=current_balance-amount
    USER_PRIVATE_CREDENITIALS[user]["balance"]=new_balance
    return {"message":f"{amount}$ withdrawn from {user}'s account successfully. Updated balance: {new_balance}$"}

@app.get("/wallet/transaction/deposit")
def deposit(user:user_dependency,logged:user_logged_dependency,amount:input_amount_dependency):
    current_balance=USER_PRIVATE_CREDENITIALS[user]["balance"]
    new_balance=current_balance+amount
    USER_PRIVATE_CREDENITIALS[user]["balance"]=new_balance
    return {"message":f"{amount}$ deposited to {user}'s account successfully. Updated balance: {new_balance}$"}
