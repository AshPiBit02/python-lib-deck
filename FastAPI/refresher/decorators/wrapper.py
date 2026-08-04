# Functions are objects

def my_decorator(func):
    def wrapper():
        print("Before the function runs")
        func()
        print("After the function runs")
    return wrapper()

@my_decorator
def say_hello():
    print("Hello!")
say_hello

# Decorating functions that take arguments

from typing import Callable,Any

def logger(func:Callable)->Callable:
    def wrapper(*args:Any,**kwargs:Any)->Any:
        print(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        result=func(*args,**kwargs)
        print(f"{func.__name__} returned {result}")
        return result
    return wrapper

@logger
def add(a:int,b:int)->int:
    return a+b
add(4,5)

@logger
def greet(name:str,age:int)->str:
    if age<18:
        return f"Hello {name}, kiddo!"
    else:
        return f"Hello {name}, Sir!"

print(greet("Aashish",21))



