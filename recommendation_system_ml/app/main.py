
from fastapi import FastAPI, HTTPException
from recommend import RecommenderAPI

app = FastAPI(title="E-Commerce Recommender API")

recommender = RecommenderAPI()

@app.get("/")
def root():
    return {"message": "Welcome to the Product Recommender API"}

@app.get("/recommend/user/{user_id}")
def recommend(user_id: str, n: int = 5):
    try:
        recommendations = recommender.recommend_for_user(user_id, top_n=n)
        return {"user_id": user_id, "recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
