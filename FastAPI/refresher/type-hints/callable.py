from typing import Callable


# Typing a function as a value
def apply_operation(a:int, b:int, operation:Callable[[int,int],int])->int:
    return operation(a,b);

def multiply(x:int,y:int)->int:
    return x*y;

result=apply_operation(3,4,multiply)
print("Callable[typing a function as a value]: ",result)


def apply_pump(a:int,operation:Callable[[int],int])->int:
    return operation(a)
def increment(x:int)->int:
    return x+1
def decrement(x:int)->int:
    return x-1

print("Increment",apply_pump(3,increment))
print("Increment",apply_pump(3,decrement))

balance=2500
def transaction(amount:int,operation:Callable[[int],None])->None:
    operation(amount)

def withdraw(amount:int):
    print("$",amount," withdrew successfully!")
    global balance 
    balance=balance-amount
    print("Updated Balance: $",balance)

def deposit(amount:int):
    print("$",amount," deposited successfully!")
    global balance 
    balance=amount+balance
    print("Updated Balance: $",balance)

transaction(500,withdraw)
transaction(700,deposit)


# Function that returns a function
def make_multiplier(factor:int)->Callable[[int],int]:
    def multiplier(x:int)->int:
        return x*factor
    return multiplier
times_four: Callable[[int],int]=make_multiplier(4)
print(times_four(7))


def num1(x:int)->Callable[[int],int]:
    def num2(y:int)->int:
        return x+y
    return num2
adder:Callable[[int],int]=num1(10)
print(adder(20))


# Practice
def average(*numbers:float)->float:
    if not numbers:
        return 0.0
    total=sum(numbers)
    return total/len(numbers)
print("Average: ",average(1.2,3.2,4,3.4))


def describe_person(**info: str) -> str:
    return ", ".join(f"{key}:{value}" for key,value in info.items())
print(describe_person(name="Aashish",university="Pokhara University",age=21))

def apply_twice(func:Callable[[int],int],value:int)->int:
    return func(func(value))

def square(num:int)->int:
    return num*num

def increment(num:int)->int:
    return num+1

# calling with a named function
print("Square: ",apply_twice(increment,5))
# calling with a Lambda
print("Lambda: ",apply_twice(lambda x:x+10,5))
# calling with callable class
class Multiplier:
    def __init__(self,factor:int):
        self.factor=factor
    def __call__(self,x:int)->int:
        return x * self.factor
double = Multiplier(2)
print("Callable class: ",apply_twice(double,5))


def make_validator(min_length:int)->Callable[[str],bool]:
    def validator(s:str)->bool:
        return len(s)>=min_length
    return validator

validator_min3=make_validator(3)
validator_min8=make_validator(8)
print("Min 3 chars, 'Hi': ",validator_min3("Hi"))
print("Min 3 chars, 'Hiiiii': ",validator_min3("Hiiiii"))
print("Min 8 chars, 'abcdefgsfa': ",validator_min8("abcdefgsfa"))

def filter_items(items:list[int],predicate: Callable[[int],bool])->list[int]:
    return [item for item in items if predicate(item)]

numbers=[1,32,81,4,28,6,7]
evens=filter_items(numbers,lambda x:x%2==0)
print("Even numbers: ",evens)

greater_than10=filter_items(numbers,lambda x:x>10)
print("Number greater than 10: ",greater_than10)

