from fastapi import FastAPI
 
from routers.customer_router import customer_router, secure_customer_router
from routers.account_router import account_router, secure_account_router
from routers.card_router import card_router, secure_card_router
from routers.transaction_router import pin_secure_transaction_router, secure_transaction_router
from routers.audit_router import audit_router
 
app = FastAPI(title="Mobile Banking API")
 
app.include_router(customer_router)
app.include_router(secure_customer_router)
 
app.include_router(account_router)
app.include_router(secure_account_router)
 
app.include_router(card_router)
app.include_router(secure_card_router)
 
app.include_router(secure_transaction_router)
app.include_router(pin_secure_transaction_router)
 
app.include_router(audit_router)
 