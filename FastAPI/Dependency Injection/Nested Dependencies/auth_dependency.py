from fastapi import FastAPI,Depends,Header,HTTPException
from typing import Annotated
app=FastAPI()

fake_users_db={
    "token-jon":{"username":"jon","role":"user"},
    "token-rob":{"username":"rob","role":"admin"},
}

def get_current_user(authorization:str=Header(...))->dict:
    user=fake_users_db.get(authorization)
    if user is None:
        raise HTTPException(status_code=401,detail="Invalid token")
    return user

current_user_dependency=Annotated[dict,Depends(get_current_user)]
def get_current_admin(user:current_user_dependency)->dict:
    if user["role"]!="admin":
        raise HTTPException(status_code=403,detail="Admins only")
    return user

current_admin_dependency=Annotated[dict,Depends(get_current_admin)]

@app.get("/profile")
def read_profile(user:current_user_dependency):
    return {"username":user["username"]}

@app.delete("/admin/purge")
def purge_data(admin:current_admin_dependency):
    return {"message":f"Purge executed by {admin['username']}"}