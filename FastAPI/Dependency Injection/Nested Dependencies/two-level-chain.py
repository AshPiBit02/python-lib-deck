from fastapi import FastAPI,Depends,HTTPException,Header
from typing import Annotated
app=FastAPI()

fake_client_db={"client-1": {"name": "Acme", "tier": "free"}, "client-2": {"name": "Globex", "tier": "premium"}}

def get_api_client(x_client_id:str=Header(...))->dict:
    client=fake_client_db.get(x_client_id)
    if client is None:
        raise HTTPException(status_code=401,detail="Invalid or missing client ID")
    return client

api_client_dependency=Annotated[dict,Depends(get_api_client)]

def require_premium_client(client:api_client_dependency)->dict:
    if client["tier"]!="premium":
        raise HTTPException(status_code=403,detail="This resource requires a premium tier subscription")
    return client

premium_client_dependency=Annotated[dict,Depends(require_premium_client)]

@app.get("/reports/basic")
def report_basic(api_client:api_client_dependency):
    return {"name":api_client["name"],"tier":api_client["tier"]}

@app.get("/reports/advanced")
def report_advanced(premium_client:premium_client_dependency):
    return {"name":premium_client["name"],"tier":premium_client["tier"],"services":"advanced"}

