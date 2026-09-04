from sqlalchemy.orm import Session
from models import AuditLog,LogStatus

def log_action(db:Session,action:str,customer_id:int|None,
               details:str|None,status:LogStatus=LogStatus.success,
               commit_independently:bool=False,
               )->AuditLog:
    entry=AuditLog(action=action,customer_id=customer_id,details=details,status=status)
    db.add(entry)
    if commit_independently:
        db.commit()
        db.refresh(entry)
    else:
        db.flush()
    return entry

def get_audit_logs(db:Session,skip:int=0,limit:int=100)->list[AuditLog]:
    return db.query(AuditLog).order_by(AuditLog.created_at.desc()).offset(skip).limit(limit).all()

def get_audit_logs_for_customer(db:Session,customer_id:int)->list[AuditLog]:
    return db.query(AuditLog).filter(AuditLog.customer_id==customer_id).order_by(AuditLog.created_at.desc()).all()

def get_failed_actions(db:Session)->list[AuditLog]:
    return db.query(AuditLog).filter(AuditLog.status==LogStatus.failed).order_by(AuditLog.created_at.desc()).all()