from pydantic import BaseModel
from fastapi import FastAPI

app=FastAPI()

class BookBase(BaseModel):
    title:str
    author:str
    copies_available:int

class BookIn(BookBase):
    internal_notes:str

class BookOut(BookBase):
    id:int

books=[]

@app.post("/books",response_model=BookOut,status_code=201)
def add_book(book:BookIn):
    new_book={
        "id":len(books)+1,
        "title":book.title,
        "author":book.author,
        "copies_available":book.copies_available,
        "internal_notes":book.internal_notes
    }
    books.append(new_book)
    return BookOut(**new_book)