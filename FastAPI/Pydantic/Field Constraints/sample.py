from pydantic import BaseModel,Field,ValidationError

class Coupon(BaseModel):
    code:str=Field(...,min_length=4,max_length=10,pattern=r"^[A-Z0-9$]+$") # field required
    discount_percent:int=Field(...,gt=0,le=100) # field required
    max_uses:int=Field(default=1,ge=1)
    items:list[str]=Field(default_factory=list,max_length=20)

coupon1=Coupon(code="SAVE$50",discount_percent=12,max_uses=3)
print(coupon1)

try:
    coupon2=Coupon(code="sv5",discount_percent=120,max_uses=2)
except ValidationError as e:
    print(e.errors())

