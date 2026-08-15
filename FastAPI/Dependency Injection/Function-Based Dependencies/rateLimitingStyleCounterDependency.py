from fastapi import FastAPI,Depends,HTTPException

app=FastAPI()

req_count=0
def check_request_count()->int:
    global req_count
    req_count+=1
    if req_count>5:
        raise HTTPException(status_code=429,detail="Too many requests")
    return req_count

@app.get("/Request")
def request_count(count:int=Depends(check_request_count)):
    return {"Current Count":req_count}
   
