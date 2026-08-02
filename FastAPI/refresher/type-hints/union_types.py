from fastapi import FastAPI
from typing import Union

app=FastAPI()

@app.get("/student/")
def student_identifier(id:Union[int,str],active:Union[bool,str]):
    return {
        "ID":id,"Active":active
    }

@app.get("/student/score")
def student_score(score: int|str):
    if isinstance(score,str):
        return {"Numeric Score":score}
    else:
        return {"Grade Letter":score}
