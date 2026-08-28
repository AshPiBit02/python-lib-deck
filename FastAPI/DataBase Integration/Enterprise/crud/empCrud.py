from sqlalchemy.orm import Session
from models.employee import Employee

def get_employees(db:Session):
    return db.query(Employee).all()

def get_employees_by_id(db:Session,emp_id:int):
    return db.query(Employee).filter(Employee.id==emp_id).first()

def add_new_employee(db:Session,emp:Employee):
    new_emp=Employee(full_name=emp.full_name,email=emp.email,position=emp.position,salary=emp.salary,is_active=emp.is_active,department_id=emp.department_id)
    db.add(new_emp)
    db.commit()
    db.refresh(new_emp)
    return new_emp
