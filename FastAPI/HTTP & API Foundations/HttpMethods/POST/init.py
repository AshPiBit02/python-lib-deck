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
    return employees

# Pattern 3: Validation/Wrong Type
class Person(BaseModel):
    name:str
    age:int

Persons=[]
@app.post("/person")
def add_persion(person:Person):
    Persons.append(person.model_dump())
    return person 

# Pattern 4: Nested Models
class Address(BaseModel):
    city:str
    country:str
class Student(BaseModel):
    name:str
    address:Address
students_addresses=[]
@app.post("/student_address")
def add_student_address(student:Student):
    students_addresses.append(student.model_dump())
    return f"{student.name} added!"






