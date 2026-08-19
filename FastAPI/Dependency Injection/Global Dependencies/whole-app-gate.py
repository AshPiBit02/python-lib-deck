from fastapi import FastAPI,Depends,Header,HTTPException
from dummydata import orders,shipments
VALID_REQ_ID=["req-1","req-2","req-3","req-4"]

def verify_request_id(x_request_id:str=Header(...))->None:
    if x_request_id not in VALID_REQ_ID:
        raise HTTPException(status_code=403,detail="Invalid request id")
    return x_request_id

app=FastAPI(dependencies=[Depends(verify_request_id)])

@app.get("/orders")
def get_orders():
    return orders

@app.get("/shipments")
def get_shipments():
    return shipments

@app.get("/audit-log")
def get_audit_log(req_id:str=Depends(verify_request_id)):
    return {"Resquest ID":req_id.upper()}

