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


# Preserving function metadata (functools.wraps) -> without it, add.__name__ becomes "wrapper" instead of "add"

from functools import wraps

def logger(func:Callable)->Callable:
    @wraps(func)
    def wrapper(*args:Any,**kwargs:Any)->Any:
        print(f"Calling {func.__name__}")
        return func(*args,**kwargs)
    return wrapper

@logger
def do_vote(name:str,age:int)->str:
    if age>18:
        return f"You can vote, {name}"
    else:
        return f"You can't vote, {name} kid"

print(do_vote(name="Jon",age=16))


# Decoratros that take their own arguments

def repeat(times:int)->Callable:
    def decorator(func:Callable)->Callable:
        @wraps(func)
        def wrapper(*args:Any,**kwargs:Any)->Any:
            for _ in range(times):
                func(*args,**kwargs)
        return wrapper
    return decorator

@repeat(times=3)
def greet(name:str)->None:
    print(f"Hi {name}")
greet("Aegon")


num = 3
@repeat(times=3)
def increment(n:None)->None:
    global num
    num=num+1
    print(f"Count: {num}")
increment(num)


# Class-based decorators

class CountCalls:
    def __init__(self,func:Callable)->None:
        self.func=func
        self.count=0

    def __call__(self,*args:Any,**kwargs:Any)->Any:
        self.count+=1
        print(f"Call #{self.count} to {self.func.__name__}")
        return self.func(*args,**kwargs)

@CountCalls
def say_hi()->None:
    print("Hi")
say_hi()
say_hi()
say_hi()
say_hi()