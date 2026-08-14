from pydantic import BaseModel,Field,EmailStr,model_validator,ConfigDict,field_validator
from typing import Literal
from datetime import date
from fastapi import FastAPI,HTTPException

app=FastAPI()
class Address(BaseModel):
    street:str
    city:str
    country:str

hotels:list[dict]=[]
rooms:list[dict]=[]
bookings:list[dict]=[]

def next_id(items:list[dict])->int:
    return max((item["id"] for item in items),default=0)+1
class Hotel(BaseModel):
    id:int|None=None
    name:str=Field(...,min_length=4,max_length=100)
    address:Address
    star_rating:int=Field(ge=1,le=5)
    contact_email:EmailStr

class RoomBase(BaseModel):
    room_number:str
    room_type:str
    price_per_night:float=Field(ge=1200,le=100000,multiple_of=0.5)
    max_occupancy:int=Field(gt=1,le=10)

class RoomIn(RoomBase):
    internal_notes:str

class RoomOut(RoomBase):
    id:int
    is_available:bool

class CardPayment(BaseModel):
    method:Literal["card"]
    card_last4:str=Field(pattern=r"^\d{4}$")

class CashPayment(BaseModel):
    method:Literal["cash"]
    amount:float=Field(gt=0)

class PayPalPayment(BaseModel):
    method:Literal["paypal"]
    paypal_email:EmailStr


class BookingBase(BaseModel):
    guest_name:str
    guest_email:EmailStr
    room_id:int
    check_int:date
    check_out:date
class BookingIn(BookingBase):
    payment:CardPayment|CashPayment|PayPalPayment=Field(...,discriminator="method")

    @field_validator("guest_name")
    @classmethod
    def guest_name_not_blank(cls,name:str)->str:
        if not name.strip():
            raise ValueError("Guest name can't be blank or whitespace!")
        return name
    @model_validator(mode="after")
    def check_date(self)->"BookingIn":
        if self.check_in>=self.check_out:
            raise ValueError("Invalid checking dates!")
        return self

    model_config=ConfigDict(
        str_strip_whitespace=True,
        extra="forbid"
    )

class UpdateBooking(BaseModel):
    guest_name:str|None=None
    guest_email:EmailStr|None=None
    check_int:date|None=None
    check_out:date|None=None
    status:Literal["confirmed","cancelled"]|None=None
    model_config=ConfigDict(extra="forbid")
class BookingOut(BookingBase):
    id:int
    payment_method:Literal["card","cash","paypal"]
    total_price:float
    status:Literal["confirmed","cancelled"]

@app.post("/hotels",response_model=Hotel,status_code=201)
def create_hotel(hotel:Hotel):
    new_id=next_id(hotels)
    stored={**hotel.model_dump(),"id":new_id}
    hotels.append(stored)
    return stored


@app.post("/hotels/{hotel_id}/rooms",response_model=RoomOut,status_code=201)
def add_room(hotel_id:int,room:RoomIn):
    if not any(h["id"]==hotel_id for h in hotels):
        raise HTTPException(status_code=404,detail=f"Hotel {hotel_id} not found!")

    new_id=next_id(rooms)
    stored={
        **room.model_dump(),
        "id":new_id,
        "hotel_id":hotel_id,
        "is_available":True
    }
    rooms.append(stored)
    return stored

@app.get("/hotels/{hotel_id}/rooms",response_model=list[RoomOut])
def list_rooms(hotel_id:int):
    if not any(h["id"]==hotel_id for h in hotels):
        raise HTTPException(status_code=404,detail=f"Hotel {hotel_id} not found!")
    return [r for r in rooms if r["hotel_id"]==hotel_id]

