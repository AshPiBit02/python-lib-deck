from fastapi import FastAPI,Depends
from typing import Annotated
app=FastAPI()

call_count={"get_db":0}

def get_db():
    call_count["get_db"]+=1
    print(f"get_db called - total calls: {call_count['get_db']}")
    return {"connection":"fake-db"}

db_dependency=Annotated[dict,Depends(get_db)]
# db_dependency=Annotated[dict,Depends(get_db,use_cache=False)] # forces get_db to run everytime it's injected, even if multiple dependencies use it in the same request

def dep_a(db:db_dependency)->dict:
    return {"from":"dep-a","db":db}

def dep_b(db:db_dependency)->dict:
    return {"from":"dep-b","db":db}

@app.get("/dep_a")
def dep_a_route(a:dict=Depends(dep_a)):
    return {"dependency":a,"total_get_db_calls":call_count["get_db"]}
@app.get("/combined")
def combined_route(a:dict=Depends(dep_a),b:dict=Depends(dep_b)):
    return {"a":a,"b":b,"total_get_db_calls":call_count["get_db"]}