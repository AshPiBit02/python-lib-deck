from typing import Literal
from fastapi import FastAPI

app=FastAPI()

@app.get("/student/grade")
def student_grade(grade:Literal["A","B","C","D","F"]):
    return {"Grade":grade,"Valid":True}

@app.get("/student/status")
def student_status(status:Literal["active","inactive","suspened"]):
    return {"Status":status}


@app.get("/student/exam")
def student(exam_type:Literal["midterm","final","quiz"]):
    return {"Exam Type":exam_type}