from pydantic import BaseModel,HttpUrl,EmailStr,ValidationError

class Applicant(BaseModel):
    full_name:str
    email:EmailStr
    portfolio_url:HttpUrl

applicant1=Applicant(full_name="Aashish Chaudhary",email="aashishchaudhari249@gmail.com",portfolio_url="https://ashipit.in")
print(applicant1)
try:
    applicant2=Applicant(full_name="Unknown man",email="unknownmangmailcom",portfolio_url="https://theunknown.in")
except ValidationError as e:
    print(e.errors())
