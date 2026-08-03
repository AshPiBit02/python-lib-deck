from typing import Callable

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
