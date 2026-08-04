import asyncio


# Core keywords
async def say_hello() -> str:
    return "Hello!"

async def main()-> None:
    result=await say_hello()
    print(result)

asyncio.run(main())

# Simulating I/O wait

import time
async def fetch_data(name:str,delay:int)->str:
    print(f"Start fetching {name}")
    await asyncio.sleep(delay)
    print(f"Done fetching {name}")
    return f"{name} data"
async def main()->None:
    start=time.perf_counter()
    result = await fetch_data("A",2)
    print(result)
    print(f"Took {time.perf_counter()-start:.4f}s")

asyncio.run(main())