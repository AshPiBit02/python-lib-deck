from fastapi import FastAPI,Depends
from sqlalchemy.orm import Session
from db.database import get_db
from typing import Annotated
import crud.user as crud
from pydantic import EmailStr

app=FastAPI()
database_dependency=Annotated[Session,Depends(get_db)]

@app.get("/users/read/{user_id}")
def read_user(user_id:int,db:database_dependency):
    user=crud.get_user(db,user_id)
    return user

@app.post("/users/add")
def create_user(user:str,email:EmailStr,db:database_dependency,is_active:bool=None):
    user=crud.create_user(db,user,email,is_active)
    return user

@app.get("/users/read")
def page_read_user(skip:int,limit:int,db:database_dependency):
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
def update_user(db:database_dependency,user_id:int,name:str=None,is_active:bool=False):
    user=crud.update_user(db,user_id,name,is_active)
    return user

