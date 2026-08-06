from fastapi import FastAPI

app = FastAPI()

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

# GET /students
@app.get("/students")
def get_students():
    return students

# GET /students/{id}
@app.get("/students/{id}")
def get_student_by_id(id:int):
    for student in students:
        if student["id"]==id:
            return student
    return {"message":f"Student not found with id {id}"}

# Filter
@app.get("/students/filter/by_dept/{department}")
def filter_students(department:str):
    results=[
    student for student in students if student["department"].lower()==department.lower()
    ]
    if results:
        return results
    else:
        return {"message":f"no student found in department {department}"}

# Search
@app.get("/students/search/by_name/{token}")
def search_student(token:str):
    result=[student for student in students if token.lower() in student["name"].lower()]
    if result:
        return result
    else: 
        return {"message":f"no student found with {token}"}

# Pagination
@app.get("/students/page/{page}/limit/{limit}")
def paginate(page:int,limit:int):
    start=(page-1)*limit
    end=start+limit
    return students[start:end]

# Sorting
@app.get("/students/sort/{order}")
def sort_students(order:str):
    if order=="asc":
        return sorted(students,key=lambda x:x["cgpa"])
    elif order=="desc":
        return sorted(students,key=lambda x:x["cgpa"],reverse=True)
    else:
        return {"message":f"unknown order {order}"}