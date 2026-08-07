from fastapi import FastAPI
from pydantic import BaseModel

app=FastAPI()

students=[]

class Student(BaseModel):
    name:str
    age:int
    department:str

# @app.post("/students")
# def create_student(student:Student):
#     students.append(student.model_dump())
#     return {
#         "message":"Student added","student":student
#     }

@app.get("/student_list")
def get_students():
    return students

# POST Patterns

# pattern 1: Create Student

# @app.post("/students")
# def create_student(student:Student):
#     student.append(student.model_dump())
#     return student # get printed in response body

# Pattern 2: Auto ID
students=[]
@app.post("/students")
def create_student(student:Student):
    new_student={
        "id":len(students)+1,
        **student.model_dump()
    }
    students.append(new_student)
    return new_student

class Employee(BaseModel):
    name:str
    role:str
    address:str
    department:str
    salary:float
    active:bool=False
employees=[]
@app.post("/add_employee")
def add_employee(employee:Employee):
    new_employee={
        "id":len(employees)+100,
        **employee.model_dump()
    }
    employees.append(new_employee)
    return{
        "message":"new employee added!",
        "details":f"id: {new_employee["id"]} | name: {new_employee["name"]} | department: {new_employee["department"]}"
    }

@app.get("/employee/list")
def employees_list():
    return employees;



