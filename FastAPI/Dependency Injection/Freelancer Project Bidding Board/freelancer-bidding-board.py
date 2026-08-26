from fastapi import FastAPI,Depends,Header,HTTPException,APIRouter
from typing import Annotated

app=FastAPI()

freelancers_db={
    "token-jon":{"username":"jon","reputation":"new","active_bids":0},
    "token-maria":{"username":"maria","reputation":"established","active_bids":2},
    "token-alex":{"username":"alex","reputation":"expert","active_bids":5},
}

projects_db=[
    {"id":1,"title":"Landing page redesign","budget":500},
    {"id":2,"title":"API integration","budget":1200},
]

bids_db:list[dict]=[]

BID_LIMITS={"new":2,"established":5}

def next_bid_id()->int:
    return max((b["id"] for b in bids_db),default=0)+1


call_count={"get_current_freelancer":0}

def get_current_freelancer(x_auth_token:str=Header(...))->dict:
    call_count["get_current_freelancer"]+=1
    print(f"get_current_freelancer() ran - call #{call_count["get_current_freelancer"]}")
    freelancer=freelancers_db.get(x_auth_token)
    if freelancer is None:
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

class BidLimitChecker:
    def __init__(self,limits:dict[str,int]):
        self.limits=limits

    def __call__(self,freelancer:freelancer_dependency)->None:
        reputation=freelancer["reputation"]
        if reputation not in self.limits:
            return
        limit=self.limits[reputation]
        if freelancer["active_bids"]>=limit:
            raise HTTPException(status_code=403,detail=f"Bid limit reached for '{reputation}' tier (max {limit} active bids)")

check_bid_limit=BidLimitChecker(BID_LIMITS)
bid_limit_dependency=Annotated[None,Depends(check_bid_limit)]


bidding_router=APIRouter(prefix="/projects",dependencies=[Depends(get_current_freelancer)])

@bidding_router.get("")
def list_projects():
    return projects_db

@bidding_router.post("/{project_id}/bid")
def place_bid(project_id:int,freelancer:freelancer_dependency,_limit_check:bid_limit_dependency,session:bid_session_dependency):
    project=next((p for p in projects_db if p["id"]==project_id),None)
    if project is None:
        raise HTTPException(status_code=404,detail=f"Project {project_id} not found")
    if not session["active"]:
        raise HTTPException(status_code=500,detail="Bid session expired!")
    new_bid={"id":next_bid_id(),"project_id":project_id,"freelancer":freelancer["username"],"amount":project["budget"],}
    bids_db.append(new_bid)
    freelancer["active_bids"]+=1
    return {"message":f"Bid placed bu {freelancer['username']} on project {project_id}",
            "bid":new_bid,
            "get_current_freelancer_calls_this_request":call_count["get_current_freelancer"],
            }

@bidding_router.get("/{project_id}/bids")
def list_project_bids(project_id:int):
    return [b for b in bids_db if b["project_id"]==project_id]

@app.get("/me/bids")
def my_bids(freelancer:freelancer_dependency):
    return [b for b in bids_db if b["freelancer"]==freelancer["username"]]

app.include_router(bidding_router)

if __name__=="__main__":
    from fastapi.testclient import TestClient

    def fake_new_freelancer_at_limit()->dict:
        return {"username":"test_new_user","reputation":"new","active_bids":2}

    app.dependency_overrides[get_current_freelancer]=fake_new_freelancer_at_limit

    client=TestClient(app)
    response=client.post("/projects/1/bid")

    print("\n--- dependency_overrides test ---")
    print("Status:",response.status_code)
    print("Body:",response.json())
    assert response.status_code==403,"Expected 403 - freelancer is already at bid limit"
    print("override test passed - blocked as expected")

    app.dependency_overrides.clear()

