from fastapi import FastAPI,Depends,HTTPException
from sqlalchemy.orm import Session
from db.database import get_db
from typing import Annotated
import crud.empCrud as Empcrud
from pydantic import EmailStr,BaseModel


app=FastAPI()
database_dependency=Annotated[Session,Depends(get_db)]

@app.get("/enterprise/employee/list")
def employee_list(db:database_dependency):
    return Empcrud.get_employees(db)
