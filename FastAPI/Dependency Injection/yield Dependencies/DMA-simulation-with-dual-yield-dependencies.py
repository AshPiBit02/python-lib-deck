from fastapi import FastAPI,Depends
import asyncio

app=FastAPI()

def get_setup():
    print("CPU: Initializes DMA controller and prepares memory buffer")
    configs={"source":"memory","destination":"I/O device","block_size":"12 bytes"}
    try:
        yield configs
    finally:
        print("DMAC: Requests control of the system bus")

async def get_bus_access():
    print("CPU: Grants system Bus access to DMAC")
    transmission={"task":"transmitting"}
    try:
        print("DMAC: Transmitting the blocks.......")
        await asyncio.sleep(2)
        yield transmission
        print("DMAC: Data transfer completed")
    finally:
        print("DMAC: Releases the system bus back to CPU")

@app.get("/file_transfer")
def transfer_file(setup:dict=Depends(get_setup),bus:dict=Depends(get_bus_access)):
    print("Route: Simulating DMA file transfer")
    return {"setup":setup,"bus":bus}

