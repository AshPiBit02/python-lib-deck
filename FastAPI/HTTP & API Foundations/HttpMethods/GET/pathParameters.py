from fastapi import FastAPI
app=FastAPI()

employees={
  "E001": { "name": "Ram Shrestha", "age": 32, "department": "IT", "salary": 55000 },
  "E002": { "name": "Sita Gurung", "age": 28, "department": "HR", "salary": 42000 },
  "E003": { "name": "Hari Koirala", "age": 45, "department": "Finance", "salary": 68000 },
  "E004": { "name": "Anil Thapa", "age": 39, "department": "Marketing", "salary": 47000 },
  "E005": { "name": "Maya Lama", "age": 26, "department": "Sales", "salary": 40000 },
  "E006": { "name": "Bikash Rai", "age": 34, "department": "Operations", "salary": 52000 },
  "E007": { "name": "Sunita KC", "age": 30, "department": "IT", "salary": 56000 },
  "E008": { "name": "Prakash Adhikari", "age": 41, "department": "Finance", "salary": 70000 },
  "E009": { "name": "Rita Magar", "age": 29, "department": "HR", "salary": 43000 },
  "E010": { "name": "Kiran Shahi", "age": 36, "department": "Sales", "salary": 48000 },
  "E011": { "name": "Laxmi Poudel", "age": 27, "department": "Marketing", "salary": 45000 },
  "E012": { "name": "Dipesh Tamang", "age": 33, "department": "Operations", "salary": 51000 },
  "E013": { "name": "Sabina Rana", "age": 31, "department": "IT", "salary": 57000 },
  "E014": { "name": "Nabin Basnet", "age": 40, "department": "Finance", "salary": 69000 },
  "E015": { "name": "Kusum Khadka", "age": 25, "department": "HR", "salary": 41000 }
}
@app.get("/")
def home():
    return "Welcome to Employees Dashboard"

@app.get("/employees")
def get_employees():
    return employees

@app.get("/employees/{emp_id}")
def get_emp_by_id(emp_id:str):
    if emp_id in employees:       
        return employees[emp_id]
    return f"Employee with emp_id {emp_id} not found!"


# Multiple Path Parameters
@app.get("/employees/department/{dept}/age/{age}")
def get_emp_by_dept_age(dept:str,age:int):
    results=[]
    for emp_id,emp in employees.items():
        if emp["department"].lower()==dept.lower() and emp["age"]<age:
            results.append({emp_id:emp})
    if results:
        return results
    return {"error":f"No employees found in {dept} under age {age}"}

# Query Parameters

@app.get("/employees")
def get_emp_with_more_salary(salary:float):
    results=[]
    for emp_id,emp in employees.items():
        if emp["salary"]>salary:
            results.append({emp_id:emp})
    if results:
        return results
    return {"error":f"No employees found found with salary more than {salary}$"}

# Optional Query Parameters
@app.get("/employees")
def empByID(emp_id:str|None=None):
    if emp_id is None:
        return employees
    if emp_id in employees:
        return employees[emp_id]
    return {"error":f"No employee found with ID {emp_id}"}