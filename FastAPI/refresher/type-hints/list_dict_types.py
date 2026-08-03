from fastapi import FastAPI
from typing import List

app = FastAPI()

@app.get("/student/marks")
def student_marks(marks:str):
    marks_list=[int(x) for x in marks.split(",")]
    return {"Marks": marks_list, "Average": sum(marks_list) / len(marks_list)}

from typing import Dict

@app.get("/student/record")
def student_record(name: str, age: int):
    record: Dict[str, str | int] = {"Name": name, "Age": age}
    if isinstance(record, dict):
        return {"Result": "Valid", "Record": record}
    else:
        return {"Result": "Invalid"}


@app.get("/student/courses")
def student_courses(courses:str):
    course_list=[int(x) for x in courses.split(",")]
    return {"Course Count":len(course_list)}

@app.get("/student/profile")
def student_profile(name: str, age: int, skills: List[str]):
    return {"Name": name, "Age": age, "Skills": skills}
