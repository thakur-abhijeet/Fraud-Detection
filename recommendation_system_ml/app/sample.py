
import pandas as pd
import random
import uuid
from faker import Faker

fake = Faker("en_NP")

brands = ["Samsung", "LG", "TCL", "Baltra", "Lava", "Ultima", "Transsion"]
categories = ["Smartphone", "TV", "Refrigerator", "Microwave", "Air Cooler", "Washing Machine"]
attributes_pool = {
    "Smartphone": ["4G", "Dual SIM", "128GB", "6GB RAM", "6.5-inch"],
    "TV": ["LED", "32 inch", "Smart TV", "HD"],
    "Refrigerator": ["Double Door", "200L", "Cooler", "Energy Star"],
    "Microwave": ["Convection", "Grill", "Solo", "700W"],
    "Air Cooler": ["35L Tank", "Remote", "Ice Chamber"],
    "Washing Machine": ["Front Load", "7kg", "Inverter"],
}

def gen_products(num=50):
    rows = []
    for _ in range(num):
        pid = str(uuid.uuid4())[:8]
        cat = random.choice(categories)
        br = random.choice(brands)
        attrs = " ".join(random.sample(attributes_pool[cat], 2))
        title = f"{br} {cat} {fake.word().capitalize()}"
        price = random.randint(10000, 100000)
        rating = round(random.uniform(2.5, 5.0), 1)
        rows.append([pid, title, br, cat, attrs, price, rating])
    df = pd.DataFrame(rows, columns=["product_id","title","brand","category","attributes","price","rating"])
    df.to_csv("data/products.csv", index=False)

def gen_ratings(products, users=20):
    rows = []
    user_ids = [f"U{1000+i}" for i in range(users)]
    for uid in user_ids:
        sampled = products["product_id"].sample(10)
        for pid in sampled:
            r = round(random.uniform(1,5),1)
            rows.append([uid, pid, r])
    pd.DataFrame(rows, columns=["user_id","product_id","rating"]).to_csv("data/user_ratings.csv", index=False)

def gen_reviews(products, reviews_per=5):
    rows = []
    for pid in products["product_id"]:
        for _ in range(reviews_per):
            text = fake.sentence(nb_words=12)
            rows.append([pid, text])
    pd.DataFrame(rows, columns=["product_id","review_text"]).to_csv("data/reviews.csv", index=False)

if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    gen_products()
    prods = pd.read_csv("data/products.csv")
    gen_ratings(prods)
    gen_reviews(prods)
    print("Sample data generated.")
