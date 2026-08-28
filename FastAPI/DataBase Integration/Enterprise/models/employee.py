from sqlalchemy import Column, Date, ForeignKey,Integer, Numeric,String,Boolean
from sqlalchemy.orm import relationship
from db.database import Base

class Employee(Base):
    __tablename__="employees"
    id=Column(Integer,primary_key=True,index=True)
    full_name=Column(String(150),nullable=False)
    email=Column(String(200),unique=True,nullable=False)
    position=Column(String(100),nullable=True)
    salary=Column(Numeric(12,2),nullable=False)
    is_active=Column(Boolean,default=True)
    department_id=Column(Integer,ForeignKey("departments.id"),nullable=True)
    
    department=relationship("Department",back_populates="employees")