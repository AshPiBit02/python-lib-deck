from sqlalchemy.orm import Session
from models import Employee,Department,HLOrder
from sqlalchemy import func
from decimal import Decimal

def dept_exists(db:Session,dept_name:str)->bool:
    dept=db.query(Department).filter(func.lower(Department.name)==dept_name.lower()).count()>0
    if not dept:
        return False
    return True
def dept_by_id(db:Session,dept_id:int):
    dept=db.query(Department).filter(Department.id==dept_id).first()
    return dept

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

def search_department_by_key(db:Session,key:str):
    depts=db.query(Department).filter(Department.name.ilike(f"%{key}%")).all()
    return depts

def update_department(db:Session,dept_id:int,updated_department:Department):
    dept=dept_by_id(db,dept_id)
    if not dept:
        return None
    if updated_department.name:
        dept.name=updated_department.name
    if updated_department.location:
        dept.location=updated_department.location
    if updated_department.budget:
        dept.budget=updated_department.budget
    db.commit()
    db.refresh(dept)
    return dept

def get_paged_department(db:Session,skip:int,limit:int):
    depts=db.query(Department).order_by(Department.id.asc()).offset(skip).limit(limit).all()
    return depts

def get_budget_by_department(db:Session,dept:str):
    budget=db.query(Department.budget).filter(func.lower(Department.name)==dept.lower()).scalar()
    return budget

def get_department_by_budget_order(db:Session,order:HLOrder):
    if order==HLOrder.high_to_low:
        depts=db.query(Department).order_by(Department.budget.desc()).all()
    else:
        depts=db.query(Department).order_by(Department.budget.asc()).all()
    return depts
