from typing import Annotated
from fastapi import FastAPI,Depends
import asyncio

app=FastAPI()

async def turn_on_device():
    print("Turning on device...")
    device={"status":"ON"}
    try:
        print("Device turned on")
        yield device
    finally:
        device["status"]="OFF"
        print("Device turned off")

async def connect_to_internet(device_on:Annotated[dict,Depends(turn_on_device)]):
    print("Connecting to wifi....")
    wifi={"connection":"Build"}
    try:
        print("Connected to wifi")
        yield wifi
    finally:
        wifi["connection"]="Closed"
        print("Disconnect from wifi")

async def text_app(internet_connection:Annotated[dict,Depends(connect_to_internet)]):
    print("Opening text app...")
    text={"message":"'dummy message that is to be sent'","receiver":"user122","status":"sending"}
    try:
        print(f"Sending message to {text['receiver']}")
        yield text
    finally:
        text["status"]="sent"
        print("Message sent sucessfully")

@app.get("/send_message")
async def send_message(text:Annotated[dict,Depends(text_app)]):
    print("Route: Using text app to send message")
    return {"result":text}
