from fastapi import FastAPI
app=FastAPI()
@app.get("/")
def hello():
    return {'message':'Hello, sir!'}
@app.get("/about")
def about():
    return {'about':'Started learning fastapi from aug 1, 2026 yes'}