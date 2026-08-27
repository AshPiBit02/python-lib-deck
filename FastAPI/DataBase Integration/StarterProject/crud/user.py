from sqlalchemy.orm import Session
from models.user import User

def get_user(db:Session,user_id:int):
    return db.query(User).filter(User.id==user_id).first()

def get_user_by_email(db:Session,email:str):
    return db.query(User).filter(User.email==email).first()

def get_users(db:Session,skip:int=0,limit:int=100):
    return db.query(User).offset(skip).limit(limit).all()

def create_user(db:Session,name:str,email:str,is_active:bool=None):
    user=User(name=name,email=email,is_active=is_active)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def update_user(db:Session,user_id:int,name=str,is_active:bool=None):
    user=db.query(User).filter(User.id==user_id).first()
    if user is None:
        return None
    user.name=name
    user.is_active=is_active
    db.commit()
    return user

def delete_user(db:Session,user_id:int)->bool:
    user=db.query(User).filter(User.id==user_id).first()
    if user is None:
        return False
    db.delete(user)
    db.commit()
    return True

