from pydantic import BaseModel
from fastapi import FastAPI

app=FastAPI()

class BookIn(BaseModel):
    title:str
    author:str
    copies_available:int
    internal_notes:str

class BookOut(BaseModel):
    id:int
    title:str
    author:str
    copies_available:int

@app.post("/books",response_model=BookOut,status_code=201)
def create_book(book:BookIn):
    saved={"id":1,**book.model_dump()}
    return saved