from sqlalchemy.orm import Session
from models.employee import Employee

def get_employees(db:Session):
    return db.query(Employee).all()
