# from fastapi import FastAPI
# from typing import Any

# app=FastAPI()

# @app.get("/student/data")
# def student_data(info:Any):
#     return {"Received":info,"Type":str(type(info).__name__)}

from fastapi import FastAPI
from typing import Any

app = FastAPI()

@app.get("/student/data")
def student_data(info: Any):
    return {"Received": info, "Type": str(type(info).__name__)}
