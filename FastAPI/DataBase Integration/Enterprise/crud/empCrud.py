from sqlalchemy.orm import Session
from models import Employee,Department

def get_employees(db:Session):
    return db.query(Employee).all()

def get_paged_employees(db:Session,skip:int,limit:int):
    return db.query(Employee).offset(skip).limit(limit).all()

def get_employees_by_id(db:Session,emp_id:int):
    return db.query(Employee).filter(Employee.id==emp_id).first()

def add_new_employee(db:Session,emp:Employee):
    new_emp=Employee(full_name=emp.full_name,email=emp.email,position=emp.position,salary=emp.salary,is_active=emp.is_active,department_id=emp.department_id)
    db.add(new_emp)
    db.commit()
    db.refresh(new_emp)
    return new_emp

def get_employee_by_dept(db: Session, dept: str):
    return (
        db.query(Employee)
        .join(Department, Employee.department_id == Department.id)
        .filter(Department.name == dept)
        .all()
    )

def get_employee_by_salary_range(db:Session,min:float,max:float):
    emp=db.query(Employee).filter(Employee.salary>=min,Employee.salary<=max).all()
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

