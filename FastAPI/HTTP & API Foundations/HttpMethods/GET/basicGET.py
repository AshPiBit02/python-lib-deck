"""
Purpose: Retrive data(read-only)
Request body: Not used(parameters go in URL or query string)
Response: Returns data(JSON,XML,HTML,etc).
Idempotency: Yes - calling multiple times doesn't change server state.
Caching: Often cached by browers/CDNs
Security notes: Sensitive data should not be sent in query strings(visible in logs/URLs).
Typical headers:
    Accept: application/json -> Client expects JSON.
    Authorization: Bearer <token> -> sercure access.
"""
from fastapi import FastAPI
app=FastAPI()

@app.get("/")
def home():
    return {"message":"Hello Sir!"}

@app.get("/fruits")
def fruits():
    return{
        "Apple","Orange","Banana"
    }

students=[{"id":1,"name":"Aegon"},{"id":2,"name":"Jon"}]
@app.get("/students")
def get_students():
    return students

@app.get("/square")
def square():
    number=7
    return{
        "number":number,"square":number*number
    }

@app.get("/profile")
def profile():
    return{
        "name":"Rhaena","age":21,"skills":["python","C++","JAVA","C","SQL"],"address":{"city":"Pokhara","country":"Nepal"}    
    }

@app.get("/city")
def get_city():
    return "Pokhara"

@app.get("/college")
def get_college():
    return "Pokhara Engineering College"

@app.get("/numbers")
def get_numbers():
    return [1,2,3,4,5]

@app.get("/employee")
def get_employee():
    return {"name":"employee 101","salary":20000,"department":"HR","skills":["decision making","leadership","fluent communication","presuasive"]}

@app.get("/books")
def get_books():
    return {"books":["Into the Wild","Atomice Habbits","48 Laws of Power"]}