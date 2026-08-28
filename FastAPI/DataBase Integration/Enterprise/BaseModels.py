from pydantic import BaseModel,Field,EmailStr,field_validator
from decimal import Decimal

class EmpAdd(BaseModel):
    full_name:str=Field(min_length=5)
    email:EmailStr
    position:str
    salary:Decimal=Field(gt=0)
    is_active:bool=True
    department_id:int|None=None

class DeptAdd(BaseModel):
    name:str=Field(min_length=5)
    location:str=Field(min_length=1)
    budget:Decimal=Field(gt=0)

class EmpAddResponse(BaseModel):
    id:int
    full_name:str
    position:str
    department_id:int

class DeptAddResponse(BaseModel):
    id:int
    name:str
    location:str

class EmpResponse(BaseModel):
    id:int
    full_name:str
    email:EmailStr
    position:str
    salary:Decimal
    is_active:bool=True
    department_id:int



        



