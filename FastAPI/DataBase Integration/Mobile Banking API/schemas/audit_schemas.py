from pydantic import BaseModel,Field,ConfigDict
from datetime import datetime
from typing import Optional
from models import LogStatus

class AuditLogBase(BaseModel):
    action:str=Field(...)
    details:Optional[str]=None
    status:LogStatus

class AuditLogCreate(AuditLogBase):
    customer_id:Optional[int]=None

class AuditLogResponse(AuditLogBase):
    id:int
    customer_id:Optional[int]=None
    created_at:datetime
    model_config=ConfigDict(from_attributes=True)

class AuditLogQuery(BaseModel):
    customer_id:Optional[int]=None
    status:Optional[LogStatus]=None
    start_date=Optional[datetime]=None
    end_date=Optional[datetime]=None

