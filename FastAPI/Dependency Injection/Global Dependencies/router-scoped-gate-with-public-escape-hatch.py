from fastapi import FastAPI,Depends,Header,HTTPException
from fastapi.routing import APIRouter
from dummydata import inventory,announcements

STAFF_TOKENS=["STF_TKN-07","STF_TKN-09","STF_TKN-33"]
def verify_staff_token(x_token:str=Header(...))->None:
    if x_token not in STAFF_TOKENS:
        raise HTTPException(status_code=403,detail="Invalid token")

internal_router=APIRouter(prefix="/internal",dependencies=[Depends(verify_staff_token)])
public_router=APIRouter(prefix="/public")

@internal_router.get("/inventory")
def view_inventory():
    return inventory

@public_router.get("/announcements")
def get_announcement():
    return announcements

app=FastAPI()

app.include_router(internal_router)
app.include_router(public_router)