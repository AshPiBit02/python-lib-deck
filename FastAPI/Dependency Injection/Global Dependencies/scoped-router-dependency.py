from fastapi import FastAPI,APIRouter,Depends,Header,HTTPException
from dummydata import private_details,public_details
app=FastAPI()

def verify_api_key(secretkey:str=Header(...)):
    if secretkey!="secretkey567":
        raise HTTPException(status_code=403,detail="Invalid API key")

admin_router=APIRouter(prefix="/admin",dependencies=[Depends(verify_api_key)])

@admin_router.get("/user_private_details")
def get_user_private_details():
    return private_details

public_router=APIRouter()

@public_router.get("/user_public_details")
def get_user_public_details():
    return public_details

app.include_router(admin_router)
app.include_router(public_router)