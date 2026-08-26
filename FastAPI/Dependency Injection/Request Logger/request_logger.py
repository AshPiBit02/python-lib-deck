from fastapi import FastAPI,HTTPException,Depends,Header
from typing import Annotated
import uuid

def verify_api_key(x_api_key:str=Header(...)):
    if x_api_key!="secret123":
        raise HTTPException(status_code=403,detail="Invalid or missing API key")
    
request_count={}
request_count["count"]=0

def request_counter()->None:
    request_count["count"]+=1

app=FastAPI(dependencies=[Depends(verify_api_key),Depends(request_counter)])

def get_request_id():
    return str(uuid.uuid4())

db={"users":{}}
def get_db():
    try:
        yield db
    finally:
        pass

@app.get("/users")
def get_users(db:dict=Depends(get_db)):
    return {"users":list(db["users"].values())}

@app.get("/users/{user_id}")
def get_user(user_id:int,db:dict=Depends(get_db),request_id:str=Depends(get_request_id)):
    user=db["users"].get(user_id)
    if not user:
        raise HTTPException(status_code=404,detail="User not found")
    return {"request_id":request_id,"user":user}

@app.get("/request_count")
def get_request_count():
    return {"Count":request_count["count"]}

@app.post("/users/{username}")
def create_user(username:str,db:dict=Depends(get_db)):
    user_id=len(db["users"])+1
    db["users"][user_id]=username
    return {"id":user_id,"user":username}
