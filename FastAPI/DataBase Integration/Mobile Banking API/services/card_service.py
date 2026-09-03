from sqlalchemy.orm import Session
from fastapi import HTTPException
from models import Card,Account
from schemas import CardCreate,CardUpdate
from sqlalchemy.exc import IntegrityError
from services.audit_service import log_action,LogStatus

def create_card(db:Session,card:CardCreate)->Card:
    new_card=Card(
        account_id=card.account_id,
        card_number=card.card_number,
        expiry_date=card.expiry_date,
        )
    customer_id=db.query(Account.customer_id).filter(Account.id==card.account_id).scalar()
    try:
        db.add(new_card)
        log_action(db,"card_creation",customer_id,f"New card: {new_card.card_number}",LogStatus.success)
        db.commit()
        db.refresh(new_card)
        return new_card
    except IntegrityError as e:
        db.rollback()
        log_action(db,"card_creation",customer_id,f"Failed to create card: {str(e)}",LogStatus.failed,commit_independently=True)
        msg=str(e.orig).lower()
        if "account_id" in msg:
            raise HTTPException(status_code=400,detail=f"Account {card.account_id} already has a card")
        elif "card_number" in msg:
            raise HTTPException(status_code=400,detail="Card number already in use")
        elif "foreign key" in msg:
            raise HTTPException(status_code=400,detail=f"Account {card.account_id} does not exists")
        else:
            raise HTTPException(status_code=400,detail="Database error")


def get_card_by_account_id(db:Session,account_id:int)->Card:
    card=db.query(Card).filter(Card.account_id==account_id).first()
    if card is None:
        raise HTTPException(status_code=404,detail=f"No card found for account {account_id}")
    return card

def update_card_status(db:Session,account_id:int,updates:CardUpdate):
    card=get_card_by_account_id(db,account_id)
    customer_id=db.query(Account.customer_id).filter(Account.id==card.account_id).scalar()
    updated_data=updates.model_dump(exclude_unset=True)
    try:
        for field,value in updated_data.items():
            setattr(card,field,value)
        log_action(db,"card_update",customer_id,f"Fields updated: {list(updated_data.keys())}",LogStatus.success)
        db.commit()
        db.refresh(card)
        return card
    except Exception as e:
        db.rollback()
        log_action(db,"card_update",customer_id,f"Failed to update card {card.card_number}: {str(e)}",LogStatus.failed,commit_independently=True)
        raise HTTPException(status_code=400,detail=f"Failed to update card {card.card_number}")


