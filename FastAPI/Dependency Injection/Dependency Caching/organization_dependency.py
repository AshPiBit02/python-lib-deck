from fastapi import FastAPI,Depends,Header,HTTPException

app=FastAPI()

fake_orgs_db={
    "org-101":{"name":"Nimbus Corp","plan":"pro","monthly_quota":1000,"used":850},
    "org-202":{"name":"Ferrotech","plan":"free","monthly_quota":100,"used":40}
    }

lookup_count={"get_current_org":0}

def get_current_org(x_ord_id:str=Header(...))->dict:
    lookup_count["get_current_org"]+=1
    print(f"get_current_org() ran - DB lookup #{lookup_count['get_current_org']} this run")
    org=fake_orgs_db.get(x_ord_id)
    if org is None:
        raise HTTPException(status_code=404,detail="Unknown organization")
    return org

def check_plan_allows_export(org:dict=Depends(get_current_org))->None:
    if org["plan"]=="free":
        raise HTTPException(status_code=403,detail="Export requires a Pro plan")

def check_quota_remaining(org:dict=Depends(get_current_org))->None:
    if org["used"]>=org["monthly_quota"]:
        raise HTTPException(status_code=429,detail="Monthly quota exceeded")

@app.get("/dashboard/export")
def export_report(org:dict=Depends(get_current_org),plan_check:None=Depends(check_plan_allows_export),quota_check:None=Depends(check_quota_remaining)):
    return {
        "org":org["name"],
        "message":"Export generated successfully",
        "get_current_org_calls_this_request":lookup_count["get_current_org"]
    }