from sqlalchemy.orm import Session
from models import Employee,Department,HLOrder,ExtremeValue,EmpAdd
from sqlalchemy import func
from decimal import Decimal

def emp_exists(db:Session,emp_id:int)->bool:
    emp=db.query(Employee).filter(Employee.id==emp_id).count()>0
    if not emp:
        return False
    return True

def get_employees(db:Session):
    return db.query(Employee).order_by(Employee.id.asc()).all()

def get_paged_employees(db:Session,skip:int,limit:int):
    return db.query(Employee).order_by(Employee.id.asc()).offset(skip).limit(limit).all()

def get_employee_by_id(db:Session,emp_id:int):
    return db.query(Employee).filter(Employee.id==emp_id).first()

def add_new_employee(db:Session,emp:EmpAdd):
    new_emp=Employee(full_name=emp.full_name,email=emp.email,position=emp.position,salary=emp.salary,is_active=emp.is_active,department_id=emp.department_id)
    db.add(new_emp)
    db.commit()
    db.refresh(new_emp)
    return new_emp

def search_employee_by_key(db:Session,key:str):
    emps=db.query(Employee).filter(Employee.full_name.ilike(f"%{key}%")).all()
    return emps


def get_employee_by_salary_range(db:Session,min:float,max:float):
    emp=db.query(Employee).filter(Employee.salary>=min,Employee.salary<=max).order_by(Employee.id.asc()).all()
    return emp

def update_employee_salary(db:Session,emp_id:int,new_salary:float):
    emp=db.query(Employee).filter(Employee.id==emp_id).first()
    if emp is None:
        return None
    old_salary=emp.salary
    emp.salary=new_salary
    db.commit()
    db.refresh(emp)
    return{
        "Old salary":old_salary,"Updated salary":emp.salary
    }

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

def remove_employee(db:Session,emp_id:int):
    emp=get_employees_by_id(db,emp_id)
    if not emp:
        return {"success":False,"error":f"Employee with id {emp_id} doesn't exists!"}
    db.delete(emp)
    db.commit()
    return{
        "success":True,
        "message":f"Records of employee with id {emp_id} delete successfully!"
    }

def deactivate_employee(db:Session,emp_id:int):
    emp=get_employees_by_id(db,emp_id)
    if not emp:
        return {"success":False,"error":f"Employee with id {emp_id} not found!"}
    if not emp.is_active:
        return {"success":True,"message":f"Employee with id {emp_id} is already deactive"}
    emp.is_active=False
    db.commit()
    db.refresh(emp)
    return {"success":True,"message":f"Employee with id {emp_id} deactivated"}

def reactivate_employee(db:Session,emp_id:int):
    emp=get_employees_by_id(db,emp_id)
    if not emp:
        return {"success":False,"error":f"Employee with id {emp_id} not found!"}
    if emp.is_active:
        return {"success":True,"message":f"Employee with id {emp_id} is already active"}
    emp.is_active=True
    db.commit()
    db.refresh(emp)
    return {"success":True,"message":f"Employee with id {emp_id} reactivated"}

def get_active_employee_list(db:Session):
    emps=db.query(Employee).filter(Employee.is_active).order_by(Employee.id.asc()).all()
    return emps

def get_employee_by_salary_order(db:Session,order:HLOrder):
    if order==HLOrder.high_to_low:
        emps=db.query(Employee).order_by(Employee.salary.desc()).all()
    else:
        emps=db.query(Employee).order_by(Employee.salary.asc()).all()
    return emps

def get_extreme_salary_employee(db:Session,extreme:ExtremeValue):
    if extreme==ExtremeValue.highest:
        emp=db.query(Employee).order_by(Employee.salary.desc()).first()
    else:
        emp=db.query(Employee).order_by(Employee.salary.asc()).first()
    return emp

def replace_employee(db:Session,emp_id:int,updated_emp:EmpAdd):
    dept=dept_by_id(db,dept_id)
    if not dept:
        return None
    dept.name=updated_dept.name
    dept.location=updated_dept.location
    dept.budget=updated_dept.budget
    db.commit()
    db.refresh(dept)
    return {"message": f"Department {dept_id} replaced successfully.", "data": dept}
    