from pydantic import BaseModel,field_validator,ValidationError
class User(BaseModel):
    username:str
    phone_no:str
    price:float

    @field_validator("username")
    @classmethod
    def username_not_blank(cls,v:str)->str:
        if not v.strip():
            raise ValueError("Username cannot be empty or whiltespace!")
        return v.strip()

    @field_validator("phone_no")
    @classmethod
    def clean_phone_no(cls,v:str)->str:
        cleaned_phone_no=v.replace("-","")
        return cleaned_phone_no

    @field_validator("price")
    @classmethod
    def valid_price(cls,v:float)->float:
        if v<0.0:
            raise ValueError("Price can't be negative!")
        return v

user1=User(username="ashpibit",phone_no="9815-21-0319",price=18.99)
print(user1.model_dump())