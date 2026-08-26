from fastapi import FastAPI,Depends
app=FastAPI()

def get_log_writer():
    log_buffer=[]
    print("Log opened: ",log_buffer)
    try:
        yield log_buffer
    finally:
        log_buffer.append("session closed")
        print("Log closed: ",log_buffer)

@app.get("/logs/latest")
def latest_logs(log_buffer:list=Depends(get_log_writer)):
    log_buffer.append("read logs")
    print("Route body executed: ",log_buffer)
    return {"logs":log_buffer}