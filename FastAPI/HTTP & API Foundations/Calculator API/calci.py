from fastapi import FastAPI, HTTPException

app = FastAPI()

history = []

@app.get("/add")
def add(x: int, y: int) -> dict:
    result = x + y
    log = {"operation": f"{x} + {y} = {result}"}
    history.append(log)
    return {"result": result}

@app.get("/subtract")
def subtract(x: int, y: int) -> dict:
    result = x - y
    log = {"operation": f"{x} - {y} = {result}"}
    history.append(log)
    return {"result": result}

@app.get("/multiply")
def multiply(x: int, y: int) -> dict:
    result = x * y
    log = {"operation": f"{x} * {y} = {result}"}
    history.append(log)
    return {"result": result}

@app.get("/divide")
def divide(x: int, y: int) -> dict:
    if y == 0:
        raise HTTPException(status_code=400, detail="Cannot divide by zero!")
    result = x / y
    log = {"operation": f"{x} / {y} = {result}"}
    history.append(log)
    return {"result": result}

@app.get("/history")
def get_history() -> list:
    return history

@app.delete("/history")
def clear_history() -> dict:
    history.clear()
    return {"message": "History cleared"}
