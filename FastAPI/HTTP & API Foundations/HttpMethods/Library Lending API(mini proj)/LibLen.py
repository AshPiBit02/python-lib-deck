from fastapi import FastAPI,HTTPException
from pydantic import BaseModel

app=FastAPI()

books = [
    {"id": 1, "title": "Clean Code", "author": "Robert Martin", "copies_available": 2, "borrowed_by": None},
    {"id": 2, "title": "The Pragmatic Programmer", "author": "Andrew Hunt", "copies_available": 1, "borrowed_by": None},
    {"id": 3, "title": "Design Patterns", "author": "GoF", "copies_available": 0, "borrowed_by": "Alice"},
]

class Booke(BaseModel):
    id:int
    title:str
    author:str
    copies_available:int
    borrowed_by:str|None

@app.get("/books",status_code=200)
def book_list():
    return books

@app.get("/books/{book_id}",status_code=200)
def book_by_id(book_id:int):
    for book in books:
        if book["id"]==book_id:
            return book
    raise HTTPException(status_code=404,detail=f"Book with id {book_id} not found!")

