from fastapi import FastAPI,Depends,Header,HTTPException,APIRouter
from typing import Annotated

app=FastAPI()

freelancer_db={
    "token-jon":{"username":"jon","reputation":"new","active_bids":0},
    "token-maria":{"username":"maria","reputation":"established","active_bids":2},
    "token-alex":{"username":"alex","reputation":"expert","active_bids":5},
}

project_db=[
    {"id":1,"title":"Landing page redesign","buget":500},
    {"id":2,"title":"API integration","buget":1200},
]

bids_db:list[dict]=[]

BID_LIMITS={"new":2,"established":5}

def next_bid_id()->int:
    return max((b["id"] for b in bids_db),default=0)+1


call_count={"get_current_freelancer":0}

def get_current_freelancer(x_auth_token:str=Header(...))->dict:
    call_count["get_current_freelancer"]+=1
    print(f"get_current_freelancer() ran - call #{call_count["get_current_freelancer"]}")
    freelancer=freelancer_db.get(x_auth_token)
    if freelancer_db is None:
        raise HTTPException(status_code=401,detail="Invalid or missing auth token")
    return freelancer

freelancer_dependency=Annotated[dict,Depends(get_current_freelancer)]

def get_bid_session():
    print("Bid session opened")
    session={"active":True}
    try:
        yield session
    finally:
        session["active"]=False
        print("Bid session closed")

bid_session_dependency=Annotated[dict,Depends(get_bid_session)]

