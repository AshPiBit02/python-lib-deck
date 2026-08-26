from fastapi import FastAPI,Depends

app=FastAPI()

def get_db_session():
    print("Setup A: DB session opened")
    db_session={"active":True}
    try:
        yield db_session
    finally:
        db_session["active"]=False
        print("Teardown A: DB session closed")

def get_lock():
    print("Setup B: Lock acquired")
    lock={"locked":True}
    try:
        yield lock
    finally:
        lock={"locked":False}
        print("Teardown B: Lock released")

@app.get("/transaction")
def transaction(db_session:dict=Depends(get_db_session),lock:dict=Depends(get_lock)):
    print("Route body: transaction running with ",db_session,lock)
    return {"db_sessoin":db_session,"lock":lock}