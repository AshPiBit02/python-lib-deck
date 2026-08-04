from typing import Callable,Any
import time

# Basic decorator
def shout(func:Callable)->Callable:
    def wrapper()->str:
        result=func()
        return result.upper()
    return wrapper

@shout
def greet()->str:
    return "hello"

print(greet())

# Decorator with *args,**kwargs

def timer(func:Callable)->Callable:
    def wrapper(*args:Any,**kwargs:Any)->Any:
        start=time.perf_counter()
        result=func(*args,**kwargs)
        end=time.perf_counter()
        print(f"{func.__name__} took {end-start:.4f} seconds")
        return result
    return wrapper

@timer
def work(duration:int=1)->str:
    time.sleep(duration)
    return "done"
print(work())
print(work(duration=2))

