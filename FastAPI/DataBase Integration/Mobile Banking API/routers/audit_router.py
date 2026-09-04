from fastapi import APIRouter,Header
from fastapi import Depends
from sqlalchemy.orm import Session
import services
import schemas
from core.dependencies import database_dependency

auditLog_router=APIRouter(prefix="/auditlog")

@auditLog_router.get("/view/list",response_model=list[schemas.AuditLogResponse])
def get_audit_logs(db:database_dependency,skip:int=0,limit:int=100):
    return services.get_audit_logs(db,skip,limit)

@auditLog_router.get("/view/{customer_id}",response_model=list[schemas.AuditLogResponse])
def get_customer_audit_log(db:database_dependency,customer_id:int):
    return services.get_accounts_for_customer(db,customer_id)

@auditLog_router.get("/view/failed",response_model=list[schemas.AuditLogResponse])
def get_failed_actions(db:database_dependency):
    return schemas.get_failed_actions(db)