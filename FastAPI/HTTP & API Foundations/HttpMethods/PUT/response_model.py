from fastapi import FastAPI,HTTPException
from pydantic import BaseModel

app=FastAPI()

iot_devices = [
    {
        "device_id": "IOT-001",
        "device_name": "Smart Thermostat",
        "firmware_version": "v2.3.1",
        "internal_code": "THERM-SEC-001",
        "secret_token": "token_abc12345",
        "location": "Living Room",
        "is_active": True
    },
    {
        "device_id": "IOT-002",
        "device_name": "Security Camera",
        "firmware_version": "v1.8.0",
        "internal_code": "CAM-SEC-002",
        "secret_token": "token_def67890",
        "location": "Front Door",
        "is_active": True
    },
    {
        "device_id": "IOT-003",
        "device_name": "Smart Light",
        "firmware_version": "v3.0.2",
        "internal_code": "LIGHT-SEC-003",
        "secret_token": "token_ghi11223",
        "location": "Bedroom",
        "is_active": False
    },
    {
        "device_id": "IOT-004",
        "device_name": "Smart Lock",
        "firmware_version": "v2.1.5",
        "internal_code": "LOCK-SEC-004",
        "secret_token": "token_jkl44556",
        "location": "Main Gate",
        "is_active": True
    },
    {
        "device_id": "IOT-005",
        "device_name": "Air Quality Sensor",
        "firmware_version": "v1.4.7",
        "internal_code": "SENSOR-SEC-005",
        "secret_token": "token_mno77889",
        "location": "Office",
        "is_active": True
    }
]
class Device(BaseModel):
    device_name:str
    firmware_version:str
    internal_code:str
    secret_token:str
    location:str
    is_active:bool

class DeviceResponse(BaseModel):
    device_id:str
    device_name:str
    firmware_version:str
    location:str
    is_active:bool


@app.put("/devices/{device_id}",status_code=201,response_model=DeviceResponse)
def change_device(device_id:str,device:Device):
    for index,iot_device in enumerate(iot_devices):
        if iot_device["device_id"]==device_id:
            new_device={
                "device_id":device_id,
                **device.model_dump()
            }
            iot_devices[index]=new_device
            return new_device
    raise HTTPException(status_code=404,detail=f"Device with id {device_id} not found!")

@app.get("/devices")
def device_list():
    return iot_devices