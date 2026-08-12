from pydantic import BaseModel
class Address(BaseModel):
    street:str
    city:str
    zip_code:str

class Borrower(BaseModel):
    name:str
    address:Address

borrower=Borrower(name="Aegon",address={"street":"North LM-35","city":"Berlin","zip_code":"BRLN-67"})
print(borrower.address.city)
print(borrower.model_dump())