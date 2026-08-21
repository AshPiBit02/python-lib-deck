from fastapi import FastAPI,Depends,HTTPException
from fastapi.testclient import TestClient
import httpx
import pytest

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



#Fake override for testing(manual override)
"""
def fake_temp(city:str)->float:
    return 30.0

app.dependency_overrides[get_live_temperature]=fake_temp

@app.get("/alerts/{city}")
def get_city_alert(city:str,alert:dict=Depends(get_alert_status)):
    return {"city":city,**alert}

#Test
client=TestClient(app)
response=client.get('/alerts/pokhara')
print(response.status_code,response.json())

app.dependency_overrides.clear()
"""

# pytest fixture(scoped, automatic cleanup)

@pytest.fixture
def client_with_extreme_heat():
    def fake_extreme_temp(city:str)->float:
        return 47.0
    app.dependency_overrides[get_live_temperature]=fake_extreme_temp
    yield TestClient(app)
    app.dependency_overrides.clear()

@pytest.fixture
def client_with_mild_weather():
    def fake_mild_temp(city:str)->float:
        return 22.0
    app.dependency_overrides[get_live_temperature]=fake_mild_temp
    yield TestClient(app)
    app.dependency_overrides.clear()

def test_extreme_heat_alert(client_with_extreme_heat):
    response=client_with_extreme_heat.get("/alerts/nepalgunj")
    assert response.status_code==200
    assert response.json()["level"]=="extreme"

def test_normal_weather_alert(client_with_mild_weather):
    response=client_with_mild_weather.get("/alerts/manang")
    assert response.status_code==200
    assert response.json()["level"]=="normal"

@app.get("/alerts/{city}")
def get_city_alert(city:str,alert:dict=Depends(get_alert_status)):
    return {"city":city,**alert}



