from fastapi import FastAPI,Depends,HTTPException
from sqlalchemy.orm import Session
from db.database import get_db
from typing import Annotated
import crud.empCrud as Empcrud
from pydantic import EmailStr,BaseModel
from BaseModels import EmpResponse

app=FastAPI()
database_dependency=Annotated[Session,Depends(get_db)]

@app.get("/enterprise/employee/list")
def employee_list(db:database_dependency):
    return Empcrud.get_employees(db)

@app.get("/enterprise/employee/{id}",response_model=EmpResponse)
def employee_by_id(db:database_dependency,id:int):
    emp=Empcrud.get_employees_by_id(db,id)
    if emp is None:
        raise HTTPException(status_code=404,detail=f"Employee with id {id} not found!")
    return emp