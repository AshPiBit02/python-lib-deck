from fastapi import FastAPI
from typing import Any

app=FastAPI()

@app.get("/student/data")
def student_data(info:Any):
    return {"Received":info,"Type":str(type(info).__name__)}


@app.get("/student/record")
def student_record(record:Any):
    if isinstance(record,dict):
        return {"Result":"Valid"}
    else:
        return {"Result":"Invalid"}