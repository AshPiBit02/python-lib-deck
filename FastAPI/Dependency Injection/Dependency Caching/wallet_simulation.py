from fastapi import FastAPI,Header,HTTPException,Depends,APIRouter
from typing import Annotated
from dummy_data import USER_PRIVATE_CREDENITIALS


current_user_log_status={"userid":None,"logged":False}

def user_auth(userid:str=Header(...),password:str=Header(...))->str:
    if userid not in USER_PRIVATE_CREDENITIALS:
        raise HTTPException(status_code=404,detail="Unknown user!")
    if USER_PRIVATE_CREDENITIALS[userid]["password"]!=password:
        raise HTTPException(status_code=401,detail="Incorrect password!")
    current_user_log_status={"userid":userid,"logged":True}
    return {"message":"Login successsful"}

def validate_pin(pin:str=Header(...)):
    user_id=current_user_log_status["userid"]
    if pin != USER_PRIVATE_CREDENITIALS[user_id]["pin"]:
        raise HTTPException(status_code=401,detail="Incorrect PIN")
    return {"message":"Transaction successful"}

pin_validation_dependency=Annotated(dict,Depends(validate_pin))
login_user_dependency=Annotated(dict,Depends(user_auth))

def user_logged()







    
app=FastAPI()
