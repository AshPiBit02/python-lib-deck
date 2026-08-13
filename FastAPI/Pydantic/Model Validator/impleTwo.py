from pydantic import BaseModel,model_validator,ValidationError
class Discount(BaseModel):
    original_price:float
    discounted_price:float

    @model_validator(mode="after")
    def check_discount(self)->"Discount":
        if not (self.original_price>self.discounted_price and self.discounted_price>=self.original_price*0.1):
            raise ValueError("Discounted price must be less than original and not more than 90% off")
        return self

Discount(original_price=799,discounted_price=90)
Discount(original_price=800,discounted_price=790)
try:
    Discount(original_price=200,discounted_price=15)
except ValidationError as e:
    print(e.errors())