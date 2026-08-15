from fastapi import FastAPI,Depends,HTTPException,Header

app=FastAPI()

def get_client_platform(x_platform:str=Header(default="web"))->str:
    return x_platform

@app.get("/dashboard")
def dashboard(platform:str=Depends(get_client_platform)):
    return f"Welcome to the {platform} dashboard!"
    
