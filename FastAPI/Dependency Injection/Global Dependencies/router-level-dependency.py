from fastapi import FastAPI,Depends,Header,HTTPException
from fastapi.routing import APIRouter

def verify_api_key(x_api_key:str=Header(...))->None:
    if x_api_key!="secret456":
        raise HTTPException(status_code=403,detail="Invalid API Key")

app=FastAPI(dependencies=[Depends(verify_api_key)])

@app.get("/status")
def get_status():
    return {"status":"ok"}

@app.get("/version")
def get_version():
    return {"version":"2.1"}