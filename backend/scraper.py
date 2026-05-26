import re
import random
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta

def extract_asin(url_or_asin: str) -> str:
    """
    Extracts 10-character Amazon ASIN from a URL or raw ASIN string.
    """
    url_or_asin = url_or_asin.strip()
    # If already a 10-char ASIN
    if re.match(r'^[A-Z0-9]{10}$', url_or_asin, re.IGNORECASE):
        return url_or_asin.upper()
        
    # Match standard Amazon URLs
    asin_match = re.search(r'/(?:dp|gp/product|product-reviews)/([A-Z0-9]{10})', url_or_asin, re.IGNORECASE)
    if asin_match:
        return asin_match.group(1).upper()
        
    return "B000000000" # fallback generic ASIN

# High-quality Mock Data Generator by Product Category
MOCK_PRODUCTS = {
    "ELECTRONICS": {
        "title": "Quantum X1 Pro Smartwatch - Midnight Black",
        "category": "Electronics",
        "overall_rating": 4.2,
        "reviews": [
            ("I love the display of this watch, it is crystal clear and very responsive. The battery life is also decent, lasting about 3 days. However, the step tracker is a bit inaccurate at times.", 4, "John Doe", 3),
            ("This smartwatch is a waste of money. The battery life is horrible, it dies in less than 8 hours, and the charger feels very cheap. I would not recommend it to anyone.", 1, "DisappointedBuyer", 10),
            ("Excellent product! The design is sleek and comfortable to wear all day. The notifications sync perfectly with my phone. Battery backup is very reliable.", 5, "Sarah K.", 1),
            ("Decent smartwatch for the price. The screen and interface are nice, but the heart rate monitor shows weird readings when I'm working out. Battery is okay.", 3, "FitnessFan", 5),
            ("Amazing battery life and very fast performance. It charges fully in just 45 minutes. The sleep tracking has helped me a lot. A solid 5-star product.", 5, "David P.", 2),
            ("Absolutely frustrating. The app keeps crashing and losing connection with my watch. The watch itself looks good, but the software is garbage.", 2, "AngryTechie", 15),
            ("The screen quality is amazing and charging is super fast. But the strap quality is very poor, it gave me a skin rash. I had to buy a replacement strap.", 3, "Alexa M.", 6),
            ("Very premium build and sleek interface. Battery lasts for days without any issues. Definitely worth the price compared to other expensive models.", 5, "Robert H.", 4),
            ("Neutral opinion. It tells time, shows messages, and has some neat watch faces. It's not outstanding, but it's not bad either. Just average.", 3, "Sam L.", 20),
            ("The touch screen is very laggy. It takes two seconds to respond to swipes. But the battery backup is great, lasting almost a week. Mixed feelings.", 3, "Jenny R.", 8)
        ]
    },
    "FOOTWEAR": {
        "title": "AeroStride Men's Lightweight Running Shoes",
        "category": "Footwear",
        "overall_rating": 4.5,
        "reviews": [
            ("Extremely comfortable for running. The cushion is soft and absorbs impact well. Fits perfectly. Highly recommend these shoes!", 5, "RunnerGuy", 2),
            ("The sole started peeling off after just two weeks of light jogging. Very disappointed with the build quality. Returning these.", 1, "QualitySeeker", 12),
            ("Very lightweight and breathable. Great for summer walks. However, there is not much arch support, so my feet hurt after long runs.", 3, "Jessica T.", 4),
            ("Beautiful design, fits like a glove. The grip is excellent on wet pavement. Got many compliments at the gym.", 5, "ActiveGymmer", 1),
            ("The sizing is way off. I ordered my normal size 10 and they feel like a 9. They are very tight and uncomfortable. The design looks nice though.", 2, "ShoeLover", 7),
            ("Excellent value for money. They feel like premium running shoes but at half the price. Grip is fantastic and they look amazing.", 5, "Peter B.", 3),
            ("They are okay. The style is good, but they feel a bit stiff. Hopefully they soften up after a few runs. Decent support.", 3, "JoggerJoe", 14),
            ("Wow! These are the most comfortable shoes I've owned. Feels like walking on clouds. Excellent grip and materials.", 5, "CloudWalker", 5),
            ("Terrible chemical smell when opened, and the color looks different than the photos. Comfort is decent, but the smell makes them unwearable.", 2, "BuyerBeware", 18),
            ("Decent cushion and grip. The mesh top is nice and breathable, but water leaks in easily when it rains. Good for dry weather only.", 4, "RainJogger", 9)
        ]
    },
    "KITCHEN": {
        "title": "BrewMaster Elite Drip Coffee Maker - 12 Cup",
        "category": "Kitchen",
        "overall_rating": 4.3,
        "reviews": [
            ("Brews delicious hot coffee quickly. The programmable timer works perfectly so I wake up to fresh coffee. Easy to clean as well.", 5, "CoffeeAddict", 1),
            ("The glass carafe is extremely fragile. It cracked on the second day while washing. Replacement is hard to find. Poor design.", 2, "SadMorning", 11),
            ("Makes great coffee, but the warming plate gets way too hot and burns the coffee if left on for more than 30 minutes. Build is cheap plastic.", 3, "BeanGrinder", 5),
            ("Very simple to use and makes a hot cup. Compact design doesn't take much counter space. Very happy with this purchase.", 5, "KitchenQueen", 2),
            ("It leaks water from the bottom reservoir every time I fill it. Absolute garbage build quality. Do not buy this leaking machine.", 1, "DryCounter", 16),
            ("Great coffee maker! Coffee temperature is perfect. The reusable filter works well and saves money. Quiet brewing process.", 5, "MorningBuzz", 4),
            ("It is average. Brews fine, but the basket overflowed once because the valve got stuck. It's okay if you watch it carefully.", 3, "CaffeineFiend", 8),
            ("The brew is quick and tastes great. But the lid doesn't close properly, and the steam leaks out of the sides. Annoying but works.", 3, "HomeCook", 9),
            ("Outstanding design, looks premium on my counter. Easy programming. Coffee is bold and flavorful.", 5, "EspressoLover", 3),
            ("Neutral. It does the job. Nothing fancy, makes coffee. The carafe handle feels a bit loose, so be careful when pouring.", 3, "OfficeUser", 15)
        ]
    }
}

def generate_mock_reviews(query: str, count: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Generates high-quality mock product and reviews based on search text keywords.
    """
    query_upper = query.upper()
    
    # Match category based on keywords
    if any(k in query_upper for k in ["SHOE", "RUNNING", "FOOTWEAR", "BOOT", "SNEAKER"]):
        product_data = MOCK_PRODUCTS["FOOTWEAR"]
    elif any(k in query_upper for k in ["COFFEE", "MUG", "BREW", "KITCHEN", "MAKER", "POT"]):
        product_data = MOCK_PRODUCTS["KITCHEN"]
    else:
        # Default to Electronics/Smartwatch
        product_data = MOCK_PRODUCTS["ELECTRONICS"]

    product_info = {
        "id": extract_asin(query),
        "title": product_data["title"],
        "category": product_data["category"],
        "overall_rating": product_data["overall_rating"],
        "url": f"https://www.amazon.com/dp/{extract_asin(query)}",
        "reviews_count": len(product_data["reviews"])
    }

    reviews_list = []
    base_date = datetime.now()

    # Generate reviews, duplicating if they request more than available
    for i in range(count):
        raw_review_idx = i % len(product_data["reviews"])
        text, rating, author, days_ago = product_data["reviews"][raw_review_idx]
        
        # Add slight variations to prevent absolute duplicates if count > length
        if i >= len(product_data["reviews"]):
            text = text + " " + random.choice(["Highly recommend!", "Pretty decent overall.", "Satisfied.", "Not the best experience."])
            author = f"{author}_{random.randint(10,99)}"
            rating = max(1, min(5, rating + random.choice([-1, 0, 1])))
            days_ago += random.randint(1, 5)

        review_date = (base_date - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        
        reviews_list.append({
            "review_id": f"R{random.randint(100000, 999999)}",
            "text": text,
            "rating": rating,
            "author": author,
            "date": review_date,
            "language": "en"
        })

    return product_info, reviews_list

def scrape_amazon_reviews(url_or_asin: str, max_reviews: int = 50) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Attempts to scrape Amazon reviews. Falls back to mock data if blocked.
    """
    asin = extract_asin(url_or_asin)
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Device-Memory": "8",
    }
    
    # Try actual scraping (will fetch page 1 first)
    url = f"https://www.amazon.com/product-reviews/{asin}/ref=cm_cr_arp_d_viewopt_sr?sortBy=recent&pageNumber=1"
    
    reviews = []
    product_title = f"Amazon Product ({asin})"
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200 and "api-services-support@amazon.com" not in response.text:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find Title
            title_el = soup.select_one('a[data-hook="product-link"]')
            if title_el:
                product_title = title_el.text.strip()
            else:
                # Check page title
                page_title = soup.find('title')
                if page_title:
                    product_title = page_title.text.replace("Amazon.com: Customer reviews:", "").strip()
            
            # Extract reviews
            review_divs = soup.select('div[data-hook="review"]')
            
            for div in review_divs:
                # Review text
                body_el = div.select_one('span[data-hook="review-body"]')
                if not body_el:
                    continue
                text = body_el.text.strip()
                
                # Rating
                rating = 3
                rating_el = div.select_one('i[data-hook="review-star-rating"] span.a-icon-alt')
                if rating_el:
                    rating_match = re.search(r'(\d+)', rating_el.text)
                    if rating_match:
                        rating = int(rating_match.group(1))
                
                # Date
                date_str = None
                date_el = div.select_one('span[data-hook="review-date"]')
                if date_el:
                    # Match dates like "on October 20, 2023" or similar
                    date_match = re.search(r'on (.+)', date_el.text)
                    if date_match:
                        try:
                            parsed_date = datetime.strptime(date_match.group(1).strip(), "%B %d, %Y")
                            date_str = parsed_date.strftime("%Y-%m-%d")
                        except Exception:
                            # Standard fallback parsed date
                            date_str = datetime.now().strftime("%Y-%m-%d")
                            
                # Author
                author = "Amazon Customer"
                author_el = div.select_one('span.a-profile-name')
                if author_el:
                    author = author_el.text.strip()
                
                review_id = div.get('id', f"R{random.randint(100000, 999999)}")
                
                reviews.append({
                    "review_id": review_id,
                    "text": text,
                    "rating": rating,
                    "author": author,
                    "date": date_str or datetime.now().strftime("%Y-%m-%d"),
                    "language": "en"
                })
                
                if len(reviews) >= max_reviews:
                    break
                    
    except Exception as e:
        print(f"Scraping encountered error: {e}. Falling back to high-quality mock reviews.")

    # Fallback to mock data if blocked or error
    if not reviews or len(reviews) == 0:
        print("Scraper was blocked or returned no reviews. Activating category-aware mock reviews.")
        return generate_mock_reviews(url_or_asin, max_reviews)

    product_info = {
        "id": asin,
        "title": product_title,
        "category": "Electronics", # Default scraped category
        "overall_rating": sum(r["rating"] for r in reviews) / len(reviews) if reviews else 4.0,
        "url": f"https://www.amazon.com/dp/{asin}",
        "reviews_count": len(reviews)
    }

    return product_info, reviews
