from pydantic import BaseModel,field_validator,ValidationError
class Book(BaseModel):
    title:str
    copies_available:int

    @field_validator("title")
    @classmethod
    def title_not_blank(cls,v:str)->str:
        if not v.strip():
            raise ValueError("Title cannot be empty or whiltespace")
        return v.strip()

    @field_validator("copies_available")
    @classmethod
    def copies_not_negative(cls,v:int)->int:
        if v<0:
            raise ValueError("copies_available cannot be negative")
        return v

try:
    book0=Book(title="Into The Wild",copies_available=3)
    book1=Book(title=" ",copies_available=-2)
except ValidationError as e:
    print("Validation failed!")
    print(e.errors())