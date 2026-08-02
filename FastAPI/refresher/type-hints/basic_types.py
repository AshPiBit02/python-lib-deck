from fastapi import FastAPI
app=FastAPI()

@app.get("/student/{id}")
def get_student(id:int,name:str,passed:bool,gpa:float):
    return {
        "ID":id,"Name":name,"Passed":passed,"GPA":gpa
    }
print(get_student(2005,"Aashish",True,3.97))