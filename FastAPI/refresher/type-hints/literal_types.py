from typing import Literal
from fastapi import FastAPI

app=FastAPI()

@app.get("/student/grade")
def student_grade(grade:Literal["A","B","C","D","F"]):
    return {"Grade":grade,"Valid":True}