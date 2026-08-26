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

