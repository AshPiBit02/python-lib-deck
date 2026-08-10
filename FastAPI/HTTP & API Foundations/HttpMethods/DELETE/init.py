from fastapi import FastAPI,HTTPException
app=FastAPI()
products = [
    {
        "id": 101,
        "name": "Laptop",
        "category": "Electronics",
        "price": 95000,
        "quantity": 10
    },
    {
        "id": 102,
        "name": "Mouse",
        "category": "Electronics",
        "price": 1500,
        "quantity": 25
    },
    {
        "id": 103,
        "name": "Keyboard",
        "category": "Electronics",
        "price": 2500,
        "quantity": 15
    }
]
@app.delete("/products/{product_id}")
def delete_product(product_id:int):
    for product in products:
        if product["id"]==product_id:
            products.remove(product)
            return {"massage":f"Product with id {product_id} removed successfully!"}
    raise HTTPException(status_code=404,detail=f"Product with product id {product_id} not found!")

@app.get("/products")
def inventory():
    return products
