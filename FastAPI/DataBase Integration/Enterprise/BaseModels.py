from pydantic import BaseModel,Field,EmailStr,field_validator,ConfigDict
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


class DeptResponse(BaseModel):
    id:int
    name:str
    location:str
    budget:Decimal
    model_config = ConfigDict(from_attributes=True)

class EmpAddResponse(BaseModel):
    id:int
    full_name:str
    position:str
    department_id:int|None=None
    model_config = ConfigDict(from_attributes=True)

class DeptAddResponse(BaseModel):
    id:int
    name:str
    location:str
    model_config = ConfigDict(from_attributes=True)

class EmpResponse(BaseModel):
    id:int
    full_name:str
    email:EmailStr
    position:str
    salary:Decimal
    is_active:bool=True
    department_id:int|None=None
    model_config = ConfigDict(from_attributes=True)

class EmpSalaryResponse(BaseModel):
    id: int
    full_name: str
    position: str
    salary: float
    department_id: int|None=None
    model_config = ConfigDict(from_attributes=True)

        



