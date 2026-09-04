from fastapi import APIRouter, Depends
import services
import schemas
from core.dependencies import database_dependency, key_validation

card_router = APIRouter(prefix="/card")
secure_card_router = APIRouter(prefix="/card", dependencies=[Depends(key_validation)])


@card_router.get("/view/{account_id}", response_model=schemas.CardResponse)
def get_card(db: database_dependency, account_id: int):
    return services.get_card_by_account_id(db, account_id)


@secure_card_router.post("/add", response_model=schemas.CardResponse)
def add_card(db: database_dependency, card: schemas.CardCreate):
    return services.create_card(db, card)


@secure_card_router.patch("/update/{account_id}", response_model=schemas.CardResponse)
def update_card(db: database_dependency, account_id: int, updates: schemas.CardUpdate):
    return services.update_card_status(db, account_id, updates)

