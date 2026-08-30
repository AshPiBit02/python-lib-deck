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