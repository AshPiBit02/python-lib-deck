from fastapi import FastAPI,Depends
from typing import Annotated

app=FastAPI()
books = [
    {"id": 1, "title": "Clean Code", "author": "Robert C. Martin", "year": 2008},
    {"id": 2, "title": "The Pragmatic Programmer", "author": "Andrew Hunt", "year": 1999},
    {"id": 3, "title": "Design Patterns", "author": "Erich Gamma", "year": 1994},
    {"id": 4, "title": "Introduction to Algorithms", "author": "Thomas H. Cormen", "year": 2009},
    {"id": 5, "title": "Python Crash Course", "author": "Eric Matthes", "year": 2015}
]

class SortParams:
    def __init__(self,sort_by:str="title",descending:bool=False):
        self.sort_by=sort_by
        self.descending=descending

sort_params_dependency_default=Annotated[SortParams,Depends(SortParams)]

@app.get("/books")
def get_books(sortparams:sort_params_dependency_default):
    if not sortparams.descending:
        sorted_books=sorted(books,key=lambda x:x[sortparams.sort_by])
    else:
        sorted_books=sorted(books,key=lambda x:x[sortparams.sort_by],reverse=True)
    return sorted_books

