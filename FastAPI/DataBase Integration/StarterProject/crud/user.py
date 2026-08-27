from sqlalchemy.orm import Session
from models.user import User

def get_user(db:Session,user_id:int):
    return db.query(User).filter(User.id==user_id).first()

def get_user_by_email(db:Session,email:str):
    return db.query(User).filter(User.email==email).first()

def get_users(db:Session,skip:int,limit:int):
    return db.query(User).offset(skip).limit(limit).all()

def create_user(db:Session,name:str,email:str,is_active:bool=None):
    user=User(name=name,email=email,is_active=is_active)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def update_user(db:Session,user_id:int,name:str|None,status:bool|None):
    user=db.query(User).filter(User.id==user_id).first()
    if not user:
        return None
    if name is not None:
        user.name=name
    if status is not None:
        user.is_active=status
    db.commit()
    db.refresh(user)
    return user

def delete_user(db:Session,user_id:int)->bool:
    user=db.query(User).filter(User.id==user_id).first()
    if user is None:
        return False
    db.delete(user)
    db.commit()
    return True

