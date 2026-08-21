from fastapi import FastAPI,Header,HTTPException,Depends,APIRouter
from typing import Annotated
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

def verify_receiver(reciever:str=Header(...))->str:
    if reciever not in USER_PRIVATE_CREDENITIALS:
        raise HTTPException(status_code=404,detail="Unknown receiver!")
    return reciever


def user_logged():
    if not current_user_log_status["logged"]:
        raise HTTPException(status_code=403,detail="Session expired!")

def validate_pin(user:"user_dependency",pin:str=Header(...)):
    if pin != USER_PRIVATE_CREDENITIALS[user]["pin"]:
        raise HTTPException(status_code=401,detail="Incorrect PIN")
    return {"message":"Transaction successful"}

def inputAmount(amount:float=Header(...))->float:
    if amount<=0.0:
        raise HTTPException(status_code=400,detail="Amount must be non-negative")
    return amount

receiver_dependency=Annotated[str,Depends(verify_receiver)]
user_dependency=Annotated[str,Depends(get_user)]
pin_validation_dependency=Annotated[dict,Depends(validate_pin)]
login_user_dependency=Annotated[dict,Depends(user_auth)]
input_amount_dependency=Annotated[float,Depends(inputAmount)]

def get_balance(user:str)->float:
    balance=USER_PRIVATE_CREDENITIALS[user]["balance"]
    return balance

def set_balance(balance:float,user:str)->None:
    USER_PRIVATE_CREDENITIALS[user]["balance"]=balance
    print(f"User: {user}   |  Balance: {balance}$")

@app.post("/wallet/login")
def login(res:login_user_dependency,user:user_dependency):
    return {"message":f"Welcome, Sir({user}). Your current balance is {USER_PRIVATE_CREDENITIALS[user]['balance']}$"}

wallet_router=APIRouter(prefix="/wallet",dependencies=[Depends(user_logged)])
@wallet_router.post("/transaction/withdraw")
def withdraw(user:user_dependency,pin:pin_validation_dependency,amount:input_amount_dependency):
    current_balance=get_balance(user)
    if current_balance<amount:
        raise HTTPException(status_code=400,detail="Insufficient Balance!")
    new_balance=current_balance-amount
    set_balance(new_balance,user)
    return {"message":f"{amount}$ withdrawn from {user}'s account successfully. Updated balance: {new_balance}$"}

@wallet_router.post("/transaction/deposit")
def deposit(user:user_dependency,pin:pin_validation_dependency,amount:input_amount_dependency):
    current_balance=get_balance(user)
    new_balance=current_balance+amount
    set_balance(new_balance,user)
    return {"message":f"{amount}$ deposited to {user}'s account successfully. Updated balance: {new_balance}$"}

@wallet_router.post("/transaction/transfer")
def transfer(sender:user_dependency,receiver:receiver_dependency,pin:pin_validation_dependency,amount:input_amount_dependency):
    if sender==receiver:
        raise HTTPException(status_code=400,detail="Self transfer is not allowed!")
    sender_current_balance=get_balance(sender)
    receiver_current_balance=get_balance(receiver)
    if sender_current_balance<amount:
        raise HTTPException(status_code=400,detail="Insufficient Balance!")
    sender_new_balance=sender_current_balance-amount
    set_balance(sender_new_balance,sender)
    receiver_new_balance=receiver_current_balance+amount
    set_balance(receiver_new_balance,receiver)
    return {"message":f"{amount}$ transferred by {sender} to {receiver}.","detail":f"Updated balance: {sender}(sender): {sender_new_balance}$ | {receiver}(receiver): {receiver_new_balance}$"}

@wallet_router.get("/balance_inquiry")
def inquire_balance(user:user_dependency):
    balance=get_balance(user)
    return {"message":f"Your current balance is {balance} $"}

@wallet_router.post("/logout")
def logout(user:user_dependency):
    current_user_log_status["userid"]=None
    current_user_log_status["logged"]=False
    return {"message":f"{user} logged out sucessfully."}

app.include_router(wallet_router)