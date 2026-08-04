from typing import Callable,Any
import time
from functools import wraps
import random

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

# Decorator with *args,**kwargs with wraps

def timer(func:Callable)->Callable:
    @wraps(func)
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
print(work.__name__)
print(work())
print(work(duration=2))


# Decorator with its own arguments

def retry(times:int)->Callable:
    def decorator(func:Callable)->Callable:
        @wraps(func)
        def wrapper(*args,**kwargs):
            for attempt in range(1,times+1):
                try:
                    result=func(*args,**kwargs)
                    print(f"Attempt {attempt} success")
                    return result
                except Exception as e:
                    print(f"Attempt {attempt} failed: {e}")
                    if attempt==times:
                        print("All attempts failed. Raising exception.")
                        raise
        return wrapper
    return decorator

@retry(times=5)
def sometimes_fails()->str:
    if random.random()<0.7:
        raise ValueError("Random failure occurred!")
    return "Success!"

print(sometimes_fails())

# Class-based decorator with state

class RateLimiter:
    def __init__(self,func:Callable,max_calls:int=3)->None:
        self.func=func
        self.count=0
        self.max_calls=max_calls

    def __call__(self,*args:Any,**kwargs:Any)->Any:
        self.count+=1
        if self.count>self.max_calls:
            return f"Rate limit exceeded"
        result=self.func(*args,**kwargs)
        return f"Pool #{self.count}: {result}"

@RateLimiter
def getConnectionPool():
    return "Pool ready"

print(getConnectionPool())
print(getConnectionPool())
print(getConnectionPool())
print(getConnectionPool())
print(getConnectionPool())


# Validating arguments via decorator

def validate_positive(func:Callable)->Callable:
    @wraps(func)
    def wrapper(*args:Any,**kwargs:Any)->Any:
        for arg in args:
            if not isinstance(arg,(int,float)):
                raise TypeError("All arguments must be numeric!")
            if isinstance(arg,(int,float)) and arg<=0:
                raise ValueError("All arguments must be positive!")
        for key,value in kwargs.items():
            if not isinstance(value,(int,float)):
                raise TypeError(f"Argument '{key}' must be numeric!")
            if value<=0:
                raise ValueError(f"Argument '{key}' must be positive!")
        return func(*args,**kwargs)
    return wrapper

@validate_positive
def calculate_area(length:float,width:float)->float:
    return length*width

print(calculate_area(3,2))
print(calculate_area(length=8,width=8))
# print(calculate_area("five",9))


# Mini FastAPI simulation(A tiny mock router)

routes:dict[str,Callable]={}
def route(path:str)->Callable:
    def decorator(func:Callable)->Callable:
        routes[path]=func # register func under path
        return func
    return decorator

@route("/hello")
def hello()->str:
    return "Hello, Sir!"

@route("/bye")
def bye()->str:
    return "Bye, Sir!"

@route("/afternoom")
def afternoon()->str:
    return "Good Afternoon, Sir!"

@route("/morning")
def morning()->str:
    return "Good Morning, Sir!"

print(routes["/hello"]())
print(routes["/morning"]())