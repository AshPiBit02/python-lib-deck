from sqlalchemy import Column,Integer,String,Boolean
from db.database import Base
from pydantic import EmailStr

class User(Base):
    __tablename__="users"

    id=Column(Integer,primary_key=True,index=True)
    name=Column(String(100),nullable=False)
    email=Column(EmailStr,unique=True,nullable=False)
    is_active=Column(Boolean,default=True)