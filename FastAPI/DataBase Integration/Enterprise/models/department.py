from sqlalchemy import Column, Integer, String, Numeric
from sqlalchemy.orm import relationship
from db.database import Base

class Department(Base):
    __tablename__ = "departments"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    location = Column(String(100), nullable=False)
    budget = Column(Numeric(12, 2), nullable=False)

    employees = relationship("Employee", back_populates="department")
