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