from fastapi import FastAPI,Depends,HTTPException
from sqlalchemy.orm import Session
from db.database import get_db
from typing import Annotated
import crud.empCrud as Empcrud
from pydantic import EmailStr,BaseModel
from BaseModels import EmpResponse,EmpAdd,EmpAddResponse,EmpSalaryResponse
from sqlalchemy.exc import IntegrityError

app=FastAPI()
database_dependency=Annotated[Session,Depends(get_db)]

@app.get("/enterprise/employee/view/list")
def employee_list(db:database_dependency):
    return Empcrud.get_employees(db)

@app.get("/enterprise/employee/view/id/{id}",response_model=EmpResponse)
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

@app.get("/enterprise/employee/view/department/{department}",response_model=list[EmpResponse])
def employee_by_dept(db:database_dependency,department:str):
    emp=Empcrud.get_employee_by_dept(db,department)
    if not emp:
        raise HTTPException(status_code=404,detail=f"No employee found in {department} department")
    return emp


@app.get("/enterprise/employee/view/salary_min_max",response_model=list[EmpSalaryResponse])
def emplyee_by_salary(db:database_dependency,min:float,max:float):
    emp=Empcrud.get_employee_by_salary_range(db,min,max)
    if not emp:
        raise HTTPException(status_code=404,detail=f"No employee found with salary in range({min},{max})")
    return emp