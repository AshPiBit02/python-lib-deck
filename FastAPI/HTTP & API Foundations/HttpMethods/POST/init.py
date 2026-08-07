from fastapi import FastAPI
from pydantic import BaseModel

app=FastAPI()

students=[]

class Student(BaseModel):
    name:str
    age:int
    department:str

@app.post("/students")
def create_student(student:Student):
    students.append(student.model_dump())
    return {
        "message":"Student added","student":student
    }

@app.get("/student_list")
def get_students():
    return students

