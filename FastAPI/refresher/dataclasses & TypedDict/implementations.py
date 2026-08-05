from dataclasses import dataclass

# Basic dataclass
@dataclass
class Book:
    title:str
    author:str
    pages:int
    available:bool=True
b1=Book("Into The Wild","James Nikol",579)
b2=Book("Atomic Habbits","James Clear",693,False)
print(b1)
print(b2)

# Equality check
b3=Book("Into The Wild","James Nikol",579)
print(b1==b3)

# Mutable default gotcha
from dataclasses import field
@dataclass 
class Cart:
    items:list[str]=field(default_factory=list)
cart1=Cart()
cart2=Cart()
cart1.items.append("This is the first cart")
print(cart1)
print(cart2)

# Frozen dataclass
@dataclass(frozen=True)
class Credentials:
    account_no:int
    balance:float
cre1=Credentials(444,983.22)
# cre1.account_no=434 # error(immutable) 