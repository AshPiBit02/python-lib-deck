from fastapi import FastAPI
from typing import Optional

app=FastAPI()

@app.get("/student/")
def get_student(name:str,nickname:Optional[str]=None):
    return{
        "Name":name,"Nickname":nickname if nickname else "No nickname provided"
    }