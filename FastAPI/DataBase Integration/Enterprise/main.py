from fastapi import FastAPI,Depends,HTTPException,APIRouter,Header,Query
from sqlalchemy.orm import Session
from db.database import get_db
from typing import Annotated
import crud.empCrud as EmpCrud
import crud.deptCrud as DeptCrud
import crud.combinedCrud as CombinedCrud
from models import EmpResponse,EmpAdd,EmpAddResponse,EmpSalaryResponse,DeptResponse,DeptAddResponse,DeptAdd,DeptUpdate,HLOrder,ExtremeValue
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

secure_employee_router=APIRouter(prefix="/employee",dependencies=[Depends(key_validation)])
secure_enterprise_router=APIRouter(prefix="/enterprise",dependencies=[Depends(key_validation)])
secure_department_router=APIRouter(prefix="/department",dependencies=[Depends(key_validation)])

@employee_router.get("/view/list",response_model=list[EmpResponse])
def employee_list(db:database_dependency):
    result = EmpCrud.get_employees(db)
    if not result:
        raise HTTPException(status_code=404,detail="No employee found!")
    return result


@employee_router.get("/view/id/{id}",response_model=EmpResponse)
def employee_by_id(db:database_dependency,id:int):
    emp=EmpCrud.get_employees_by_id(db,id)
    if emp is None:
        raise HTTPException(status_code=404,detail=f"Employee with id {id} not found!")
    return emp

@secure_employee_router.post("/add/newEmployee", response_model=EmpAddResponse)
def add_employee(db: database_dependency, emp: EmpAdd):
    try:
        return EmpCrud.add_new_employee(db, emp)
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

@enterprise_router.get("/view/department/{department}",response_model=list[EmpResponse])
def employee_by_dept(db:database_dependency,department:str):
    emp=CombinedCrud.get_employee_by_dept(db,department)
    if not emp:
        raise HTTPException(status_code=404,detail=f"No employee found in {department} department")
    return emp

@employee_router.get("/view/search/nameKey",response_model=list[EmpResponse])
def employee_by_key(db:database_dependency,key:str):
    emps=EmpCrud.search_employee_by_key(db,key)
    if not emps:
        raise HTTPException(status_code=404,detail=f"No employees were found matching the search key '{key}'.")
    return emps

@employee_router.get("/view/salary_min_max",response_model=list[EmpSalaryResponse])
def empolyee_by_salary(db:database_dependency,min:float,max:float):
    emp=EmpCrud.get_employee_by_salary_range(db,min,max)
    if not emp:
        raise HTTPException(status_code=404,detail=f"No employee found with salary in range({min},{max})")
    return emp

@secure_employee_router.patch("/update/salary")
def update_employee_salary(db:database_dependency,emp_id:int,new_salary:float):
    result=EmpCrud.update_employee_salary(db,emp_id,new_salary)
    if not result:
        raise HTTPException(status_code=400,detail="Invalid employee ID or salary amount")
    return result

@employee_router.get("/view/page",response_model=list[EmpResponse])
def paged_employee(db:database_dependency,skip:int=0,limit:int=10):
    emp=EmpCrud.get_paged_employees(db,skip,limit)
    if not emp:
        raise HTTPException(status_code=404, detail=f"No departments found with offset {skip} and limit {limit}.")
    return emp

@secure_enterprise_router.patch("/update/EmployeeDepartment")
def update_employee_department(db:database_dependency,emp_id:int,new_department:str):
    result=CombinedCrud.change_employee_department(db,emp_id,new_department)
    if not result["success"]:
        raise HTTPException(status_code=400,detail=result["error"])
    return {"message":result["message"]}

@secure_employee_router.delete("/delete/ByEmployeeID")
def delete_employee(db:database_dependency,emp_id:int):
    result=EmpCrud.remove_employee(db,emp_id)
    if not result["success"]:
            raise HTTPException(status_code=400,detail=result["error"])
    return {"message":result["message"]}

@secure_employee_router.patch("/update/status/deactivate")
def deactivate_employee(db:database_dependency,emp_id:int):
    result=EmpCrud.deactivate_employee(db,emp_id)
    if not result["success"]:
        raise HTTPException(status_code=404,detail=result["error"])
    return {"message":result["message"]}

@secure_employee_router.patch("/update/status/reactivate")
def reactivate_employee(db:database_dependency,emp_id:int):
    result=EmpCrud.reactivate_employee(db,emp_id)
    if not result["success"]:
        raise HTTPException(status_code=404,detail=result["error"])
    return {"message":result["message"]}

@employee_router.get("/view/active/list",response_model=list[EmpResponse])
def active_employee_list(db:database_dependency):
    result=EmpCrud.get_active_employee_list(db)
    if not result:
        raise HTTPException(status_code=404,detail="No active employee found!")
    return result

@secure_enterprise_router.patch("/update/EmployeeSalary/ByDepartment")
def update_salary_by_department(db:database_dependency,department:str,percentage:float):
    result=CombinedCrud.update_department_salary(db,department,percentage)
    return result

@employee_router.get("/view/BySalaryOrder",response_model=list[EmpResponse])
def view_employee_by_salary_order(db:database_dependency,order:HLOrder=Query(default=HLOrder.high_to_low)):
    emps=EmpCrud.get_employee_by_salary_order(db,order)
    if not emps:
        raise HTTPException(status_code=404,detail="No employees exists. Please add departments first.")
    return emps

@employee_router.get("/view/ExtremeSalary",response_model=EmpResponse)
def view_extreme_salary_emplyee(db:database_dependency,extreme:ExtremeValue):
    emp=EmpCrud.get_extreme_salary_employee(db,extreme)
    if not emp:
        raise HTTPException(status_code=404,detail=f"No employee exits!")
    return emp

# Department Routers
@department_router.get("/view/available")
def department_exists(db:database_dependency,dept:str):
    result=DeptCrud.dept_exists(db,dept)
    if not result:
        return {"exists":False}
    return {"exists":True}

@department_router.get("/view/departmentByID",response_model=DeptResponse)
def get_department_by_id(db:database_dependency,dept_id:int):
    dept=DeptCrud.dept_by_id(db,dept_id)
    if not dept:
        raise HTTPException(status_code=404,detail=f"Deparment having id '{dept_id}' doesn't exists!")
    return dept

@department_router.get("/detailedList",response_model=list[DeptResponse])
def get_department_detailed_list(db:database_dependency):
    result=DeptCrud.dept_detail_list(db)
    if not result:
        raise HTTPException(status_code=404,detail="No departments are currently registered in the system!")
    return result

@department_router.get("/nameList")
def get_department_name_list(db:database_dependency):
    depts=DeptCrud.dept_name_list(db)
    if not depts:
        raise HTTPException(status_code=404,detail="No department names found. Please add departments first.")
    return {"Departments":depts}


@department_router.get("/view/search/nameKey",response_model=list[DeptResponse])
def search_department_by_key(db:database_dependency,key:str):
    depts=DeptCrud.search_department_by_key(db,key)
    if not depts:
        raise HTTPException(status_code=404, detail=f"No employees were found matching the search key '{key}'.")
    return depts

@secure_department_router.post("/add/newDepartment",response_model=DeptAddResponse)
def add_new_department(db:database_dependency,new_dept:DeptAdd):
    try:
        return DeptCrud.add_new_department(db,new_dept)
    except IntegrityError as e:
        db.rollback()
        print("RAW ERROR:", str(e.orig))
        msg = str(e.orig).lower()
        if "name" in msg:
            raise HTTPException(status_code=400, detail="Duplicate department not allowed")
        elif "null value" in msg:
            raise HTTPException(status_code=400, detail="Missing required field")
        else:
            raise HTTPException(status_code=400, detail="Database error")

@secure_department_router.patch("/update",response_model=DeptResponse)
def update_department(db:database_dependency,dept_id:int,dept:DeptUpdate):
    try:
        return DeptCrud.update_department(db,dept_id,dept)
    except IntegrityError as e:
        db.rollback()
        print("RAW ERROR:", str(e.orig))
        msg = str(e.orig).lower()
        if "name" in msg:
            raise HTTPException(status_code=400, detail="Duplicate department not allowed")
        else:
            raise HTTPException(status_code=400, detail="Database error")

@department_router.get("/view/pagedDeparment",response_model=list[DeptResponse])
def view_paged_department(db:database_dependency,skip:int=0,limit:int=10):
    results=DeptCrud.get_paged_department(db,skip,limit)
    if not results:
        raise HTTPException(status_code=404, detail=f"No departments found with offset {skip} and limit {limit}.")
    return results

@department_router.get("/view/budget")
def view_department_budget(db:database_dependency,department:str):
    budget=DeptCrud.get_budget_by_department(db,department)
    if not budget:
        raise HTTPException(status_code=404,detail=f"No department found with name '{department}'")
    return {"Budget":f"${budget}"}

@department_router.get("/view/ByBudgetOrder",response_model=list[DeptResponse])
def view_department_by_budget_order(db:database_dependency,order:HLOrder=Query(default=HLOrder.high_to_low)):
    depts=DeptCrud.get_department_by_budget_order(db,order)
    if not depts:
        raise HTTPException(status_code=404,detail="No department exists. Please add departments first.")
    return depts

@secure_department_router.delete("/remove/ByID")
def delete_department(db:database_dependency,dept_id:int):
    result=DeptCrud.remove_department(db,dept_id)
    if not result["success"]:
        raise HTTPException(status_code=result["code"],detail=result["error"])
    return {"message":result["message"]}

@department_router.get("/view/ExtremeBudget",response_model=DeptResponse)
def view_extreme_budget_deparment(db:database_dependency,extreme:ExtremeValue=Query(default=ExtremeValue.highest)):
    result=DeptCrud.get_extreme_budget_department(db,extreme)
    if not result:
        raise HTTPException(status_code=404,detail="No department exists!")
    return result

enterprise_router.include_router(employee_router)
enterprise_router.include_router(department_router)
enterprise_router.include_router(secure_employee_router)
enterprise_router.include_router(secure_department_router)
app.include_router(enterprise_router)