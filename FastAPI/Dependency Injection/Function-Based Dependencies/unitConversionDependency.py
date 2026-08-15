from fastapi import FastAPI,Depends,HTTPException

app=FastAPI()

UNITS=["celsius","fahrenheit"]

fakeTemp=22

def get_temp_unit(unit:str="celsius")->str:
    unit=unit.lower()
    if unit not in UNITS:
        raise HTTPException(status_code=400,detail=f"Invalid Unit '{unit}'!")
    return unit

@app.get("/weather")
def get_temp(unit:str=Depends(get_temp_unit)):
    if unit=="celsius":
        return {"Current temperature":f"{fakeTemp}°C"}
    else:
        converted=round(float(fakeTemp)*(9/5)+32,2)
        return {"Current temperature":f"{converted}°F"}
