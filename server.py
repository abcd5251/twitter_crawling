import os
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from utils import get_twitter_score

from pydantic import BaseModel

import uvicorn

import pandas as pd 

app = FastAPI()

# middleware
app.add_middleware(
    CORSMiddleware, 
    allow_credentials=True, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

class single_respond(BaseModel):
    user_name : str

@app.post("/score_respond/")
async def single_respond(item: single_respond):
    respond = get_twitter_score(item.user_name)
    print(respond)
    return {"code":0, "data":respond}

    

if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)