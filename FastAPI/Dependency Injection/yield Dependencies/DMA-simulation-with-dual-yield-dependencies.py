from fastapi import FastAPI,Depends

app=FastAPI()

def get_setup():
    print("CPU: Allocate memory buffer and configures the DMAC")
    configs={"source":"memory","destination":"I/O device","block_size":"12 bytes"}
    try:
        yield configs
    finally:
        print("DMAC: Asks for Bus System")

def get_bus_access():
    print("CPU: Grants System Bus access to DMAC")
    transmission={"task":"transmitting"}
    try:
        yield transmission
        print("Transmission completed")
    finally:
        print("DMAC: Releases the System bus")

@app.get("/file_transfer")
def transfer_file(setup:dict=Depends(get_setup),bus:dict=Depends(get_bus_access)):
    print("DMA simulation")
    return {"setup":setup,"bus":bus}

