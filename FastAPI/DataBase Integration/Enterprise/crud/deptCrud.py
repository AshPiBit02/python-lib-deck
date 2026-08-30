from sqlalchemy.orm import Session
from models import Employee,Department
from sqlalchemy import func
from decimal import Decimal

def dept_exists(db:Session,dept_name:str)->bool:
    dept=db.query(Department).filter(func.lower(Department.name)==dept_name.lower()).count()>0
    if not dept:
        return False
    return True

def dept_detail_list(db:Session):
    dept=db.query(Department).all()
    return dept

def dept_name_list(db:Session):
    depts=db.query(Department.name).all()
    return [row[0] for row in depts]

def add_new_department(db:Session,dept:Department):
    new_dept=Department(name=dept.name,location=dept.location,budget=dept.budget)
    db.add(new_dept)
    db.commit()
    db.refresh(new_dept)
    return new_dept