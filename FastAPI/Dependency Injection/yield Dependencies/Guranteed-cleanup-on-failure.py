from fastapi import FastAPI,Depends,HTTPException

app=FastAPI()
class FakeResource:
    def __init__(self):
        self.active=True
        print("Resource acquired (active=True)")

    def deactivate(self):
        self.active=False
        print("Resource released (active=False)")

def get_resource():
    resource=FakeResource()
    try:
        yield resource
    finally:
        resource.deactivate()

@app.get("/risky-task")
def risky_task(resource:FakeResource=Depends(get_resource)):
    raise HTTPException (status_code=500,detail="forced failure")
    
    