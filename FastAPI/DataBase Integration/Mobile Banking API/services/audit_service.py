from sqlalchemy.orm import Session
from models import AuditLog,LogStatus

def log_action(db:Session,action:str,customer_id:int|None,
               detials:str|None,status:LogStatus=LogStatus.success,
               commit_independently:bool=False,
               )->AuditLog:
    entry=AuditLog(action=action,customer_id=customer_id,detials=detials,status=status)
    db.add(entry)
    if commit_independently:
        db.commit()
        db.refresh(entry)
    else:
        db.flush()
    return entry