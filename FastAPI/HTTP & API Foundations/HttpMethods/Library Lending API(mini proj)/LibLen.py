from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from typing import Optional

app=FastAPI()

books = [
    {"id": 1, "title": "Clean Code", "author": "Robert Martin", "copies_available": 2, "borrowed_by": []},
    {"id": 2, "title": "The Pragmatic Programmer", "author": "Andrew Hunt", "copies_available": 3, "borrowed_by": []},
    {"id": 3, "title": "Design Patterns", "author": "GoF", "copies_available": 0, "borrowed_by": ["Alice"]},
]

class Book(BaseModel):
    title:str
    author:str
    copies_available:int
    borrowed_by:str|None

class UpdateBook(BaseModel):
    title:str|None=None
    author:str|None=None
    copies_available:int|None=None
    borrowed_by:str|None=None

@app.get("/books",status_code=200)
def book_list():
    return books

@app.get("/books/{book_id}",status_code=200)
def book_by_id(book_id:int):
    for book in books:
        if book["id"]==book_id:
            return book
    raise HTTPException(status_code=404,detail=f"Book with id {book_id} not found!")

@app.get("/books/find/{author}",status_code=200)
def book_by_author(author:str):
    result=[book for book in books if author.lower() in book["author"].lower()]
    if result:
        return result
    return {"message":f"No book found found for author {author}"}

@app.get("/books/available/{available}",status_code=200)
def available_books(available:bool):
    if available:
        result=[book for book in books if book["copies_available"]>0]
        return result
    else:
        result=[book for book in books if book["copies_available"]<1]
        return result

@app.post("/books/new_boook",status_code=201)
def add_newBook(book:Book):
    new_book={"id":len(books)+1,**book.model_dump()}
    books.append(new_book)
    return {
        "message":f"Book {book.title} added to library!"
    }

@app.put("/books/{book_id}",status_code=200)
def replace_book(book_id:int,book:Book):
    for index,available_book in enumerate(books):
        if available_book["id"]==book_id:
            new_book={
                "id":available_book["id"],
                **book.model_dump()
            }
            books[index]=new_book
            return {
                "message":f"{available_book["title"]} replaced by {book.title} successfully!"
            }
    raise HTTPException(status_code=404,detail=f"Book with id {book_id} not found!")


@app.patch("/books/{book_id}",status_code=200)
def update_book(book_id:int,book:UpdateBook):
    for existing_book in books:
        if existing_book["id"]==book_id:
            updated_book=book.model_dump(exclude_unset=True)
            existing_book.update(updated_book)
            return {"message":f"Book with book id {book_id} updated successfully!"}
    raise HTTPException(status_code=404,detail=f"Book with book id {book_id} not found!")


@app.delete("/books/{bood_id}",status_code=200)
def remove_book(book_id:int):
    for book in books:
        if book["id"]==book_id:
            books.remove(book)
            return {"message":f"Book with id {book_id} removed from library!"}
    raise HTTPException(status_code=404,detail=f"Book with id {book_id} doesn't exists!")

@app.post("/books/{book_id}/borrow/{borrower}",status_code=200)
def borrow_book(book_id:int,borrower:str):
    for existing_book in books:
        if existing_book["id"]==book_id:
            if existing_book["copies_available"]>0:
                existing_book["borrowed_by"].append(borrower)
                existing_book["copies_available"]-=1
                return {"message":f"Book with id {book_id} borrowed by {borrower} successfully!"}
            return {"message":"No copies available!"}
    raise HTTPException(status_code=404,detail=f"Book with id {book_id} doesn't exists!")

@app.post("/books/{book_id}/returner",status_code=200)
def return_book(book_id:int,returner:str):
    for existing_book in books:
        if existing_book in books:
            if existing_book["id"]==book_id:
                if returner.lower() in existing_book["borrowed_by"].lower():
                    existing_book["borrowed_by"].lower().remove(returner.lower())
                    existing_book["copies_available"]+=1
                    return {"message":f"Book with id {book_id} is return by {returner}"}
            return {"message":f"Return failed unknown returner!"}
    raise HTTPException(status_code=404,detail=f"Book with id {book_id} doesn't exists!")
