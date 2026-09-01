from sqlalchemy.orm import Session
from models import Employee,Department,AggFunc
from crud.empCrud import get_employees_by_id
from sqlalchemy import func
from decimal import Decimal

def get_employee_by_dept(db: Session, dept: str):
    return (
        db.query(Employee)
        .join(Department, Employee.department_id == Department.id)
        .filter(func.lower(Department.name) == dept.lower())
        .order_by(Employee.id.asc())
        .all()
    )

def update_department_salary(db:Session,department:str,percentage:float):
    change="increased" if percentage>0 else "decreased"
    emps=db.query(Employee).join(Department,Employee.department_id==Department.id).filter(func.lower(Department.name)==department.lower()).all()
    if not emps:
        return {"message":f"No employees found in department '{department}'"}
    factory=Decimal(1)+(Decimal(percentage)/Decimal(100))
    for emp in emps:
        emp.salary=factory*emp.salary
    db.commit()
    return {
        "message":f"Salaries of employees in department {department} {change} by {abs(percentage)}%.",
        "updated_ids":[emp.id for emp in emps]}

def change_employee_department(db: Session, emp_id: int, new_department: str):
    emp = get_employees_by_id(db,emp_id)
    if not emp:
        return {"success":False,"error": f"Employee with id {emp_id} not found!"}

    old_department_id = emp.department_id

    new_dept_id = db.query(Department.id).filter(func.lower(Department.name) == new_department.lower()).scalar()
    if new_dept_id is None:
        return {"success":False,"error": f"Department '{new_department}' not found!"}

    emp.department_id = new_dept_id
    db.commit()
    db.refresh(emp)

    return {
        "success":True,
        "message": f"Changed department of employee with id {emp_id} "
                   f"from {old_department_id} to {new_dept_id}"
    }

def total_salary_per_department(db:Session,agg:AggFunc):
    if agg==AggFunc.total:
        func_to_use=func.sum
        salary_type="total_salary"
    else:
        func_to_use=func.avg
        salary_type="average_salary"
    results=(
        db.query(Department.name,func(Employee.salary).label("total_salary"))
        .join(Employee,Department.id==Employee.department_id)
        .group_by(Department.id)
        .all()
    )

    formatted=[{"department":name,salary_type:salary} for name,salary in results]
    return formatted
