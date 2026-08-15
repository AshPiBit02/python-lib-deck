from fastapi import FastAPI,Depends,HTTPException
from dummydata import movies
app=FastAPI()

VALID_FIELDS=["title","rating","year"]
def get_sort_order(sort_by:str="title")->str:
    if sort_by not in VALID_FIELDS:
        raise HTTPException(status_code=400,detail=f"Unknown field {sort_by} cannot sort!")
    return sort_by
@app.get("/movies")
def home():
    return "Welcome Sir, Please select a movie!"

@app.get("/movies/list")
def movies_list():
    return movies

@app.get("/movies/sort_by")
def sort_movies(field:str=Depends(get_sort_order)):
    result=movies
    if field.lower()=="title":
        result=sorted(result,key=lambda m:m["title"])
    if field.lower()=="rating":
        result=sorted(result,key=lambda m:m["rating"])
    if field.lower()=="year":
        result=sorted(result,key=lambda m:m["year"])
    return result