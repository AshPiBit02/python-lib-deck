from sqlalchemy.orm import Session
from models import Employee,Department

def emp_exists(db:Session,emp_id:int)->bool:
    emp=db.query(Employee).filter(Employee.id==emp_id).count()>0
    if not emp:
        return False
    return True

def get_employees(db:Session):
    return db.query(Employee).order_by(Employee.id.asc()).all()

def get_paged_employees(db:Session,skip:int,limit:int):
    return db.query(Employee).order_by(Employee.id.asc()).offset(skip).limit(limit).all()

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
        .order_by(Employee.id.asc())
        .all()
    )

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

    new_dept_id = db.query(Department.id).filter(Department.name == new_department).scalar()
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
        return {"success":False,"error":f"Employee with id {emp_id} not found!"}
    db.delete(emp)
    db.commit()
    return{
        "success":True,
        "message":f"Records of employee with id {emp_id} delete successfully!"
    }



    
    