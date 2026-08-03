from fastapi import FastAPI
from typing import List

app = FastAPI()

@app.get("/student/marks")
def student_marks(marks:str):
    marks_list=[int(x) for x in marks.split(",")]
    return {"Marks": marks_list, "Average": sum(marks_list) / len(marks_list)}
