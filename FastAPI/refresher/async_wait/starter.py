import asyncio


# Core keywords
async def say_hello() -> str:
    return "Hello!"

async def main()-> None:
    result=await say_hello()
    print(result)

# asyncio.run(main())

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

# asyncio.run(main())

# sequential - awaiting one at a time
async def main_sequential()->None:
    start=time.perf_counter()
    await fetch_data("A",2)
    await fetch_data("B",2)
    await fetch_data("C",2)
    print(f"Total(sequential): {time.perf_counter()-start:.4f}")
# asyncio.run(main_sequential())

# concurrent - asyncio.gather():
async def main_concurrent()->None:
    start=time.perf_counter()
    results=await asyncio.gather(fetch_data("A",2),fetch_data("B",2),fetch_data("C",2))
    print(results)
    print(f"Total(concurrent): {time.perf_counter()-start:.4f}s")
# asyncio.run(main_concurrent())

# create_task for "fire now, await later":
async def main1()->None:
    start=time.perf_counter()
    task_a=asyncio.create_task(fetch_data("A",2))
    task_b=asyncio.create_task(fetch_data("B",2))
    result_a=await task_a
    result_b=await task_b
    print(f"Time: {time.perf_counter()-start:.4f}s")
# asyncio.run(main1())


