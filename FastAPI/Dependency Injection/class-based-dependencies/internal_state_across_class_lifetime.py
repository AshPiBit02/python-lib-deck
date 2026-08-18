from fastapi import FastAPI,Depends
from typing import Annotated

app=FastAPI()

class VisitCounter:
    def __init__(self):
        self.count=0

    def __call__(self)->int:
        self.count+=1
        return self.count
visit_counter_instance=VisitCounter()
counter_dependency=Annotated[int,Depends(visit_counter_instance)]
@app.get("/counter")
def get_count(count:counter_dependency):
    if count>9:
        return {"message":"Too many requests"}
    return {"Count":count}

