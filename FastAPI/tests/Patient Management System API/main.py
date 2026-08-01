from fastapi import FastAPI
app=FastAPI();

@app.get("/")
def home():
    return{"message":"Patient Management System API"}

@app.get("/about")
def about():
    return{"message":"A fully functional API to manage your patient records."}