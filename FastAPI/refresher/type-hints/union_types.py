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
    try:
        numeric_score=int(score)
        return {"Numeric Score":numeric_score}
    except(ValueError,TypeError):
        return {"Grade Letter":score}
