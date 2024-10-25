from fastapi import FastAPI
import uvicorn, nest_asyncio
from mangum import Mangum

nest_asyncio.apply()
app = FastAPI()

# from statistical import router as statistical_router
# app.include_router(statistical_router, prefix='/api/v1/statistical')
from llm_evaluation import router as llm_router
app.include_router(llm_router, prefix='/api/v1/llm-score')

from llm_recall_precision import router as llm_recall_router
app.include_router(llm_recall_router, prefix='/api/v1/llm-score')

handler = Mangum(app)

@app.get('/')
async def read_root():
    return {"message": "Welcome to the LLM Metrics API!"}

def run_app():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    run_app()
else:
    print("else")

