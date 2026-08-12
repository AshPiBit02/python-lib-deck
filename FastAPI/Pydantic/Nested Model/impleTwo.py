from pydantic import BaseModel,ValidationError
from typing import List
class Review(BaseModel):
    reviewer:str
    rating:int

class Book(BaseModel):
    title:str
    reviews:List[Review]=[]
book_ok=Book(title="The Dragon's Path",reviews=[{"reviewer":"Notang","rating":5},{"reviewer":"Batin","rating":4}])

try:
    book_bad=Book(title="The Path Dragon's",reviews=[{"reviewer":"Babu Lal","rating":3},{"reviewer":"Lulli","rating":"five"}])
except ValidationError as e:
    print(e.errors())

print(book_ok.reviews)