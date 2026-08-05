import asyncio
import time
# Basic coroutine
async def get_greeting(name:str)->str:
    return f"Hello, {name}!"
async def main()->None:
    result=await get_greeting("Sir")
    print(result)
# asyncio.run(main())

# Simulated delay
async def fetch_user(user_id:int)->dict:
    print(f"Fetching user data for user with userId: {user_id}")
    await asyncio.sleep(1)
    return {"id":user_id,"name":f"User{user_id}"}

async def main2()->None:
    start=time.perf_counter()
    result=await fetch_user(49)
    print(f"Result: {result}")
    print(f"Tool {time.perf_counter()-start:.4f}s")
# asyncio.run(main2())

# Sequential vs concurrent 
async def fetch_item(item_id:int,delay:float)->str:
    print(f"fetching item with item_ID:{item_id}")
    await asyncio.sleep(delay)
    return f"fetched successful for item_ID:{item_id}"

async def main3()->None:
    print(await fetch_item(512,1))
    print(await fetch_item(513,1))
    print(await fetch_item(514,1))
    print(await fetch_item(515,1))
    print(await fetch_item(516,1))
asyncio.run(main3())
