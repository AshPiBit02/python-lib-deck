from typing import Callable

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
