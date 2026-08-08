# HTTPException is for error that happen after the request has passed basic validation
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
app=FastAPI()

students = [
    {
        "id": 1,
        "name": "Ram",
        "age": 21,
        "department": "Computer",
        "cgpa": 3.82
    },
    {
        "id": 2,
        "name": "Hari",
        "age": 22,
        "department": "Civil",
        "cgpa": 3.45
    },
    {
        "id": 3,
        "name": "Sita",
        "age": 20,
        "department": "Computer",
        "cgpa": 3.91
    },
    {
        "id": 4,
        "name": "Gita",
        "age": 21,
        "department": "Electrical",
        "cgpa": 3.55
    },
    {
        "id": 5,
        "name": "Krishna",
        "age": 23,
        "department": "Computer",
        "cgpa": 3.20
    }
]
class Student(BaseModel):
    id:int=Field(gt=0)
    name:str=Field(min_length=1)
    age:int=Field(ge=18)
    department:str=Field(min_length=1)
    cgpa:float=Field(gt=2)


@app.post("/students",status_code=201)
def add_student(student:Student):
    students.append(student.model_dump())
    return {
        "message":f"{student.name}({student.id} added successfully!)"
    }


@app.get("/students/{id}")
def student_by_id(id:int):
    for student in students:
        if student["id"]==id:
            return student
    raise HTTPException(
        status_code=404,
        detail=f"Product with id {id} not found!"
    )
