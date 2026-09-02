from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy import func
from models import Account,AccountCustomer,Customer,Transaction,OwnerRole
from schemas import AccountCreate,AccountUpdate,JointOwnerAdd
from services import

class AccountService:
    def __init__(self,db:Session):
        self.db=db


    def create_account(self,payload:AccountCreate)->Account:
        customer=self.db.query(Customer).filter(Customer.id==payload.customer_id).first()
        if not customer:
            raise ValueError(f"Customer with id {payload.customer_id} does not exist")
        existing = self.db.query(Account).filter(Account.account_number==payload.account_number).first()
        if existing:
            raise ValueError(f"Account number {payload.account_number} already exists")

        account=Account(account_number=payload.account_number,
                        account_type=payload.account_type,
                        customer_id=payload.customer_id,
                        )

        self.db.add(account)
        self.db.commit()
        self.db.refresh(account)

        link=AccountCustomer(
            account_id=account.id,
            customer_id=payload.customer_id,
            role=OwnerRole.primary,
        )
        self.db.add(link)
        self.db.commit()

        # AuditService()
        return account

    