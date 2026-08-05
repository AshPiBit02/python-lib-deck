import asyncio
import time
import random
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

async def main3_serial()->None:
    print(await fetch_item(512,1))
    print(await fetch_item(513,1))
    print(await fetch_item(514,1))
    print(await fetch_item(515,1))
    print(await fetch_item(516,1))
# asyncio.run(main3_serial())

async def main3_concurrent()->None:
    results=await asyncio.gather(fetch_item(512,1),fetch_item(513,1),fetch_item(514,1),fetch_item(515,1),fetch_item(516,1))
    for r in results:
        print(r)
# asyncio.run(main3_concurrent())

# create_task()
async def main3_create_task()->None:
    start=time.perf_counter()
    taskA=asyncio.create_task(fetch_item(512,1.5))
    taskB=asyncio.create_task(fetch_item(513,1.5))
    taskC=asyncio.create_task(fetch_item(514,1.5))
    taskD=asyncio.create_task(fetch_item(515,1.5))
    taskE=asyncio.create_task(fetch_item(516,1.5))

    resultA=await taskA
    print(resultA)
    resultB=await taskB
    print(resultB)
    resultC=await taskC
    print(resultC)
    resultD=await taskD
    print(resultD)
    resultE=await taskE
    print(resultE)
    print(f"Time took: {time.perf_counter()-start:.4f}")
# asyncio.run(main3_create_task())

# Handling results as they complete
async def main4()->None:
    tasks=[fetch_item(100,3),fetch_item(101,1),fetch_item(102,2)]

    for completed in asyncio.as_completed(tasks):
        result=await completed
        print(result)
# asyncio.run(main4())

# Error handling in concurrent calls
async def good_task(name:str,delay:float)->str:
    await asyncio.sleep(delay)
    return f"{name} finished after {delay}s"

async def bad_task(name:str,delay:float)->str:
    await asyncio.sleep(delay)
    raise ValueError(f"{name} failed after {delay}s")

async def main5()->None:
    coros=[good_task("Task A",1),bad_task("Task B",2),good_task("Task C",3)]
    print("-"*5," With return_exception=True","-"*5)
    results = await asyncio.gather(*coros,return_exceptions=True)
    for r in results:
        print("Result: ",r)

    print("-"*5," With return_exception","-"*5)
    try:
        results=await asyncio.gather(*coros)
        print(results)
    except Exception as e:
        print("Caught exception:",e)

# asyncio.run(main5())    


# Mini weather aggregator with decorator

from functools import wraps
def async_timer(func):
    @wraps(func)
    async def wrapper(*args,**kwargs):
        start=time.perf_counter()
        result=await func(*args,**kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end-start:.4f}s")
        return result
    return wrapper
@async_timer
async def fetch_weather(city:str)->dict:
    delay=random.uniform(0.5,2)
    await asyncio.sleep(delay)
    return {"city":city,"temp":random.randint(15,35)}

async def compare_weather(cities:list[str])->list[dict]:
    start=time.perf_counter()
    results=await asyncio.gather(*(fetch_weather(city) for city in cities))
    end=time.perf_counter()
    print(f"Total time: {end-start:.4f}s")
    return results

async def main6():
    cities=["London","Paris","Tokyo","Kathmandu"]
    returns= await compare_weather(cities)
    for r in returns:
        print(r)
asyncio.run(main6())



