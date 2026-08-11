from fastapi import FastAPI,HTTPException

app=FastAPI()

sales_records = [
    {"sale_id": 1, "product_name": "Dell Laptop", "unit_price": 95000, "quantity": 2, "discount": 5000, "total": 185000},
    {"sale_id": 2, "product_name": "Nike Shoes", "unit_price": 1500, "quantity": 3, "discount": 200, "total": 4300},
    {"sale_id": 3, "product_name": "Samsung Phone", "unit_price": 80000, "quantity": 1, "discount": 0, "total": 80000},
    {"sale_id": 4, "product_name": "Office Chair", "unit_price": 7000, "quantity": 4, "discount": 500, "total": 27300},
    {"sale_id": 5, "product_name": "Acer Monitor", "unit_price": 30000, "quantity": 2, "discount": 1000, "total": 59000},
    {"sale_id": 6, "product_name": "MSI Laptop", "unit_price": 92000, "quantity": 1, "discount": 2000, "total": 90000},
    {"sale_id": 7, "product_name": "Wooden Desk", "unit_price": 12000, "quantity": 2, "discount": 500, "total": 23500},
    {"sale_id": 8, "product_name": "Sony Headphones", "unit_price": 5000, "quantity": 3, "discount": 300, "total": 14700},
    {"sale_id": 9, "product_name": "Adidas Sneakers", "unit_price": 2500, "quantity": 2, "discount": 200, "total": 4800},
    {"sale_id": 10, "product_name": "LG Refrigerator", "unit_price": 65000, "quantity": 1, "discount": 5000, "total": 60000},
]

@app.delete("/sales/{sale_id}")
def delete_record(sale_id:int):
    for index,record in enumerate(sales_records):
        if record["sale_id"]==sale_id:
            deleted_record=sales_records.pop(index)
            return {
                "message":"Sales record deleted successfully!",
                "record":deleted_record
            }

    raise HTTPException(status_code=404,detail=f"Sales record with id {sale_id} not found!")

@app.get("/sales")
def sales_record():
    return sales_records