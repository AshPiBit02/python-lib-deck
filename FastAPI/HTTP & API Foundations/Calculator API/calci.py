from fastapi import FastAPI,HTTPException
app=FastAPI()

history=[]

@app.get("/add")
def add(x:int,y:int)->dict:
    log={
        "operation":f"{x} + {y} = {x+y}"
    }
    history.append(log)
    return {"result":x+y}

@app.get("/subtract")
def sub(x:int,y:int)->dict:
    log={
        "operation":f"{x} - {y} = {x-y}"
    }
    history.append(log)
    return {"result":x-y}

@app.get("/multiply/{x}/{y}")
def mul(x:int,y:int)->dict:
    log={
        "operation":f"{x} * {y} = {x*y}"
    }
    history.append(log)
    return {"result":x*y}

@app.get("/divide/{x}/{y}")
def mul(x:int,y:int)->dict:
    if y==0:
        raise HTTPException(status_code=400,detail=f"Cannot divide by zero!")
    log={
        "operation":f"{x} / {y} = {x/y}"
    }
    history.append(log)
    return {"result":x/y}

@app.get("/history")
def history()->list:
    return history


@app.delete("/history")
def clear_history()->dict:
    history.clear()
    return {"message":"History cleared"}