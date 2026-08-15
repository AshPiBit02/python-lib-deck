from fastapi import FastAPI,Depends,HTTPException
app=FastAPI()
menu_items:list[dict]=[
     {"id": 1, "name": "Spring Rolls", "price": 6.5, "category": "starter"},
    {"id": 2, "name": "Grilled Salmon", "price": 22.0, "category": "main"},
    {"id": 3, "name": "Chocolate Cake", "price": 8.0, "category": "dessert"},
    {"id": 4, "name": "Iced Tea", "price": 3.5, "category": "drink"},
    {"id": 5, "name": "Caesar Salad", "price": 11.0, "category": "starter"}
]

ALLOWED_CATEGORIES={"starter","main","dessert","drink"}
