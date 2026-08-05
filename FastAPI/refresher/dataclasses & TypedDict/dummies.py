"""
Dataclasses & TypedDict
 both let to define structured data shapes in Python, but they solve different probelms:
i. @dataclass - a real class with actual runtime behavior(auto __init__,__repr__,__eq__,mutatble instances).Use when you need an actual object with methods/identity.
ii. TypedDict - just tells the type checker "this plain dict has these exact keys with these exact types. "No runtime behavior at all - it is a dict, nothing more.
"""

from dataclasses import dataclass,field

# @dataclass - the basic
@dataclass
class Student:
    name:str
    grade:int
    passed:bool=False
# s=Student(name="Jon",grade=93,passed=True)
# print(s)
# print(s.name)
# print(s==Student("Jon",93,True))
# print(s==Student("Jon",90,True))

# extra features
@dataclass
class Student:
    name:str
    grade:int
    passed:bool=False
    tags:list[str]=field(default_factory=list)
# s1=Student("Jon",93,True)
# s2=Student("Aegon",39)
# s1.tags.append("honor-roll")
# print(s1)
# print(s1.tags)
# print(s2.tags)

@dataclass(frozen=True)
class Point:
    x:int
    y:int
p=Point(1,2)
# p.x=2 # raises FrozenInstanceError - immutable,frozen makes instances immutable


# TypedDict - the basics
from typing import TypedDict

class StudentDict(TypedDict):
    name:str
    grade:int
    passed:bool
s:StudentDict={"name":"Aegon","grade":95,"Passed":True}
# s1:StudentDict{"name":"Daemon","grade":"ninety-nine","Passed":True} # no runtime validation
# print(s["name"])

class StudentDict(TypedDict,total=False): # total=False -> no need to give value to all fileds
    name:str
    grade:int
s:StudentDict={"name":"Aegon","grade":"two"}
# print(s["grade"])

from typing import NotRequired
class StudentDict(TypedDict):
    name:str
    grade:NotRequired[int]
s:StudentDict={"name":"Snow","grade":3.3}
# print(s["grade"])

# Converting between them

from dataclasses import asdict

@dataclass
class Student:
    name:str
    grade:int
s=Student("Alicent",89)
d:dict=asdict(s) # dataclass to dict
print(d["name"])

s2=Student(**d) # unpack dict into keyword args
print(s2)




