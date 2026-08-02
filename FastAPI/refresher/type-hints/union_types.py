from fastapi import FastAPI
from typing import Union

app=FastAPI()

@app.get("/student/")
def student_identifier(id:Union[int,str],active:Union[bool,str]):
    return {
        "ID":id,"Active":active
    }