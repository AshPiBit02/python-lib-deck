from fastapi import FastAPI,Depends,HTTPException,APIRouter,Header
from sqlalchemy.orm import Session
from db.database import get_db
from typing import Annotated
import crud.empCrud as Empcrud
import crud.deptCrud as DeptCrud
from BaseModels import EmpResponse,EmpAdd,EmpAddResponse,EmpSalaryResponse,DeptResponse
from sqlalchemy.exc import IntegrityError
from core.config import settings

app=FastAPI()
enterprise_router=APIRouter(prefix="/enterprise")
employee_router=APIRouter(prefix="/employee")
department_router=APIRouter(prefix="/department")

database_dependency=Annotated[Session,Depends(get_db)]

def key_validation(key:str=Header(...)):
    if key!=settings.secret_key:
        raise HTTPException(status_code=403,detail="Invalid secret key!")


# Employee routers

secure_router=APIRouter(prefix="/employee",dependencies=[Depends(key_validation)])

@employee_router.get("/view/list",response_model=list[EmpResponse])
def employee_list(db:database_dependency):
    result = Empcrud.get_employees(db)
    if not result:
        raise HTTPException(status_code=404,detail="No employee found!")
    return result


@employee_router.get("/view/id/{id}",response_model=EmpResponse)
def employee_by_id(db:database_dependency,id:int):
    emp=Empcrud.get_employees_by_id(db,id)
    if emp is None:
        raise HTTPException(status_code=404,detail=f"Employee with id {id} not found!")
    return emp

@secure_router.post("/add", response_model=EmpAddResponse)
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

@employee_router.get("/view/department/{department}",response_model=list[EmpResponse])
def employee_by_dept(db:database_dependency,department:str):
    emp=Empcrud.get_employee_by_dept(db,department)
    if not emp:
        raise HTTPException(status_code=404,detail=f"No employee found in {department} department")
    return emp


@employee_router.get("/view/salary_min_max",response_model=list[EmpSalaryResponse])
def empolyee_by_salary(db:database_dependency,min:float,max:float):
    emp=Empcrud.get_employee_by_salary_range(db,min,max)
    if not emp:
        raise HTTPException(status_code=404,detail=f"No employee found with salary in range({min},{max})")
    return emp

@secure_router.patch("/update/salary")
def update_employee_salary(db:database_dependency,emp_id:int,new_salary:float):
    result=Empcrud.update_employee_salary(db,emp_id,new_salary)
    if not result:
        raise HTTPException(status_code=400,detail="Invalid employee ID or salary amount")
    return result

@employee_router.get("/view/page",response_model=list[EmpResponse])
def paged_employee(db:database_dependency,skip:int,limit:int):
    emp=Empcrud.get_paged_employees(db,skip,limit)
    if not emp:
        raise HTTPException(status_code=404,detail="No employee found!")
    return emp

@secure_router.patch("/update/department")
def update_employee_department(db:database_dependency,emp_id:int,new_department:str):
    result=Empcrud.change_employee_department(db,emp_id,new_department)
    if not result["success"]:
        raise HTTPException(status_code=400,detail=result["error"])
    return {"message":result["message"]}

@secure_router.delete("/delete/employee")
def delete_employee(db:database_dependency,emp_id:int):
    result=Empcrud.remove_employee(db,emp_id)
    if not result["success"]:
            raise HTTPException(status_code=400,detail=result["error"])
    return {"message":result["message"]}

@secure_router.patch("/update/status/deactivate")
def deactivate_employee(db:database_dependency,emp_id:int):
    result=Empcrud.deactivate_employee(db,emp_id)
    if not result["success"]:
        raise HTTPException(status_code=404,detail=result["error"])
    return {"message":result["message"]}

@secure_router.patch("/update/status/reactivate")
def reactivate_employee(db:database_dependency,emp_id:int):
    result=Empcrud.reactivate_employee(db,emp_id)
    if not result["success"]:
        raise HTTPException(status_code=404,detail=result["error"])
    return {"message":result["message"]}

@employee_router.get("/view/active/list",response_model=list[EmpResponse])
def active_employee_list(db:database_dependency):
    result=Empcrud.get_active_employee_list(db)
    if not result:
        raise HTTPException(status_code=404,detail="No active employee found!")
    return result

@secure_router.patch("/update/salary/department")
def update_salary_by_department(db:database_dependency,department:str,percentage:float):
    result=Empcrud.update_department_salary(db,department,percentage)
    return result

# Department Routers
@department_router.get("/available")
def department_exists(db:database_dependency,dept:str):
    result=DeptCrud.dept_exists(db,dept)
    if not result:
        return {"exists":False}
    return {"exists":True}

@department_router.get("/detailedList",response_model=list[DeptResponse])
def get_department_detailed_list(db:database_dependency):
    result=DeptCrud.dept_detail_list(db)
    if not result:
        raise HTTPException(status_code=404,detail="No department exits!")
    return result

@department_router.get("/nameList")
def get_department_name_list(db:database_dependency):
    depts=DeptCrud.dept_name_list(db)
    if not depts:
        raise HTTPException(status_code=404,detail="No department exits!")
    return {"Departments":depts}

enterprise_router.include_router(employee_router)
enterprise_router.include_router(department_router)
enterprise_router.include_router(secure_router)
app.include_router(enterprise_router)