from fastapi import FastAPI,Depends,HTTPException
import httpx

app=FastAPI()

def get_live_temperature(city:str)->float:
    response=httpx.get(f"https://fake-weather-api.example.com/temp?city={city}")
    if response.status_code!=200:
        raise HTTPException(status_code=502,detail="Weather service unavailable")
    return response.json()["temp_celsius"]

def get_alert_status(temp:float=Depends(get_live_temperature))->dict:
    if temp>=40:
        return {"level":"extreme","advice":"Stay indoors, avoid heat exposure"}
    elif temp>=30:
        return {"level":"warning","advice":"Stay hydrated"}
    return {"level":"normal","advice":"No action needed"}

@app.get("/alerts/{city}")
def get_city_alert(city:str,alert:dict=Depends(get_alert_status)):
    return {"city":city,**alert}