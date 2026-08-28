from fastapi import FastAPI,Depends,HTTPException
from sqlalchemy.orm import Session
from db.database import get_db
from typing import Annotated
import crud.empCrud as Empcrud
from pydantic import EmailStr,BaseModel

class UserCreate(BaseModel):
    name:str
    email:EmailStr
    is_active:bool=True

class UserOut(BaseModel):
    id:int
    name:str
    email:EmailStr
    is_active:bool

    model_config={"from_attributes":True}

app=FastAPI()
database_dependency=Annotated[Session,Depends(get_db)]

@app.get("/users/read/{user_id}")
def read_user(user_id:int,db:database_dependency):
    user=crud.get_user(db,user_id)
    return user

@app.post("/users/add",response_model=UserOut)
def create_user(user:UserCreate,db:database_dependency):
    return crud.create_user(db,user.name,user.email,user.is_active)

@app.get("/users/read")
def page_read_user(db:database_dependency,skip:int=0,limit:int=100):
    user=crud.get_users(db,skip,limit)
    return user

@app.get("/users/read/by_email")
def read_user_by_email(email:EmailStr,db:database_dependency):
    user=crud.get_user_by_email(db,email)
    return user

@app.delete("/users/delete")
def delete_user(user_id:int,db:database_dependency):
    user=crud.delete_user(db,user_id)
    return user

@app.patch("/users/update")
def update_user(db:database_dependency,user_id:int,name:str|None=None,status:bool|None=None):
    user=crud.update_user(db,user_id,name,status)
    return user

