from fastapi import FastAPI,Depends
import asyncio

app=FastAPI()

class FakeConnection:
    def __init__(self):
        self.open=True
        print("Connection opened")

    async def process_query(self):
        print("Processing Query....")
        await asyncio.sleep(2)
        print("Query done!")

    def close(self):
        self.open=False
        print("Connection closed")

def get_db():
    conn=FakeConnection()
    try:
        yield conn
    finally:
        conn.close()

@app.get("/ping")
async def get_connection(conn:FakeConnection=Depends(get_db)):
    await conn.process_query()
    return {"Connection opened":conn.open}