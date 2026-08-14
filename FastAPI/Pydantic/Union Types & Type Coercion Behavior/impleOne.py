from pydantic import BaseModel,Field,EmailStr,ValidationError
from typing import Union,Literal

class EmailChannel(BaseModel):
    type:Literal["email"]
    address:EmailStr
class SmsChannel(BaseModel):
    type:Literal["sms"]
    phone:str
class Notification(BaseModel):
    channel:Union[EmailChannel,SmsChannel]=Field(...,discriminator="type")

emailChnl=Notification(channel={"type":"email","address":"aashishchadhari249@gmail.com"})
smsChnl=Notification(channel={"type":"sms","phone":"97466XXXXX"})
print(emailChnl)
print(smsChnl)

try: 
    unknown_channel=Notification(channel={"type":"push","address":"KLG-89 ST-09"})
except ValidationError as e:
    print(e.errors())