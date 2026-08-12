from pydantic import BaseModel
class Publisher(BaseModel):
    name:str
    country:str

class Book(BaseModel):
    title:str
    author:str
    publisher:Publisher

book=Book(title="Clean Code",author="Robert Martin",publisher={"name":"Prentic Hall","country":"USA"})

print(book.publisher.country)
print(book.model_dump())