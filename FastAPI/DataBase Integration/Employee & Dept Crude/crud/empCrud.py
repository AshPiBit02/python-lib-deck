from sqlalchemy.orm import Session
from models.department import Department
from models.employee import Employee

def get_user(db:Session,emp_id:int):
    return db.query(Employee).filter(Employee.id==emp_id).first()
