from sqlalchemy.orm import Session
from models.employee import Employee

def get_employees(db:Session):
    return db.query(Employee).all()

def get_employees_by_id(db:Session,emp_id:int):
    return db.query(Employee).filter(Employee.id==emp_id).first()

