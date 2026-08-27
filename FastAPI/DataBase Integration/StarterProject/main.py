from fastapi import FastAPI,Depends
from sqlalchemy.orm import Session
from db.database import get_db
from typing import Annotated
import crud.user as crud
from pydantic import EmailStr

app=FastAPI()
database_dependency=Annotated[Session,Depends(get_db)]

@app.get("/users/{user_id}")
def read_user(user_id:int,db:database_dependency):
    user=crud.get_user(db,user_id)
    return user

@app.post("/users/")
def create_user(user:str,email:EmailStr,db:database_dependency):
    user=crud.create_user(db,user,email)
    return user
