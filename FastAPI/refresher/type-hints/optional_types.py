from fastapi import FastAPI
from typing import Optional

app=FastAPI()

@app.get("/student/")
def get_student(name:str,nickname:Optional[str]=None):
    return{
        "Name":name,"Nickname":nickname if nickname else "No nickname provided"
    }

@app.get("/student/details/")
def get_details(name:str,age:Optional[int]=None,email:Optional[str]=None):
    return{
        "Name":name,"Age":age if age else "Age not provided","Email":email if email else "Email not provided"
    }