from fastapi import FastAPI,Depends,HTTPException
from sqlalchemy.orm import Session
from db.database import get_db
from typing import Annotated
import crud.empCrud as Empcrud
from pydantic import EmailStr,BaseModel
from BaseModels import EmpResponse,EmpAdd,EmpAddResponse
from sqlalchemy.exc import IntegrityError

app=FastAPI()
database_dependency=Annotated[Session,Depends(get_db)]

@app.get("/enterprise/employee/view/list")
def employee_list(db:database_dependency):
    return Empcrud.get_employees(db)

@app.get("/enterprise/employee/view/{id}",response_model=EmpResponse)
def employee_by_id(db:database_dependency,id:int):
    emp=Empcrud.get_employees_by_id(db,id)
    if emp is None:
        raise HTTPException(status_code=404,detail=f"Employee with id {id} not found!")
    return emp

@app.post("/enterprise/employee/add", response_model=EmpAddResponse)
def add_employee(db: database_dependency, emp: EmpAdd):
    try:
        return Empcrud.add_new_employee(db, emp)
    except IntegrityError as e:
        db.rollback()
        msg = str(e.orig).lower()
        if "foreign key" in msg:
            raise HTTPException(status_code=400, detail="Invalid department_id")
        elif "email" in msg:
            raise HTTPException(status_code=400, detail="Duplicate email not allowed")
        elif "null value" in msg:
            raise HTTPException(status_code=400, detail="Missing required field")
        else:
            raise HTTPException(status_code=400, detail="Database error")