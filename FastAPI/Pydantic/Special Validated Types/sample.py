from pydantic import BaseModel,EmailStr,HttpUrl,SecretStr

class Account(BaseModel):
    email:EmailStr
    website:HttpUrl
    password:SecretStr

acc=Account(email="aashishchadhari249@gmail.com",website="https://ashpibit.com",password="bitpiash")
print(acc)
print(acc.password.get_secret_value())