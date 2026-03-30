import os
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
import pandas as pd
import uuid
import time

# Automatically create the folder if it does not exist
output_dir = os.path.join("test_datasets")
os.makedirs(output_dir, exist_ok=True)

options = Options()
options.add_argument("--headless=new")
options.add_argument("--disable-gpu")
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("--log-level=3")

# Avoid bot detection 
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

driver = Chrome(options=options) # type: ignore

movies = {
    "Adipurush": "tt12915716",
    "Snow White": "tt6208148",
    "Fantastic Four": "tt1502712",
    "The Marvels": "tt10676048",
    "Eternals": "tt9032400",
    "The Shawshank Redemption": "tt0111161",
    "The Dark Knight": "tt0468569",
    "Avengers Endgame": "tt4154796",
    "Baahubali 2": "tt4849438",
    "Joker": "tt7286456",
    "Interstellar": "tt0816692",
    "Tenet": "tt6723592",
    "Doctor Strange Multiverse of Madness": "tt9419884",
    "Batman v Superman": "tt2975590",
    "Kalki 2898 AD": "tt12735488",
}

for movie_name, movie_id in movies.items():
    print(f"\n--- Scraping {movie_name} ---")
    reviews_data = []
    driver.get(f"https://www.imdb.com/title/{movie_id}/reviews?sort=submissionDate&dir=desc")
    target_reviews = 1000
    
    # Interact with the Page to load 1000 reviews
    while True:
        # Check how many reviews are currently loaded in the browser
        current_reviews = driver.find_elements(By.CSS_SELECTOR, "article.user-review-item")
        print(f"Loaded {len(current_reviews)}/{target_reviews} reviews into the page...", end="\r")
        
        if len(current_reviews) >= target_reviews:
            print(f"\nTarget reached for {movie_name}!")
            break
            
        try:
            # Look specifically for the "See all" button within the pagination container
            button_xpath = "//div[@data-testid='tturv-pagination']//button[.//span[text()='See all' or contains(text(), 'more')]]"
            
            load_btn = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.XPATH, button_xpath))
            )
            
            # Scroll to it and click via JavaScript to avoid interception errors
            driver.execute_script("arguments[0].scrollIntoView(true);", load_btn)
            time.sleep(1) 
            driver.execute_script("arguments[0].click();", load_btn)
            
            # Give the page a moment to fetch and render the massive batch of new HTML
            time.sleep(3) 
            
        except TimeoutException:
            print(f"\nNo more pagination buttons found. Reached the end of available reviews. ({len(current_reviews)} total)")
            break
        except Exception as e:
            print(f"\nStopped clicking due to an error: {e}")
            break

    print("Extracting and cleaning data...")
    
    # Pass the fully loaded HTML to BeautifulSoup
    soup = BeautifulSoup(driver.page_source, "html.parser")
    reviews = soup.find_all("article", class_="user-review-item")
    
    movie_review_count = 0
    seen_reviews = set()

    # Extract Data
    for review in reviews:
        if movie_review_count >= target_reviews:
            break

        text_tag = review.find("div", class_="ipc-html-content-inner-div")
        text = text_tag.text.strip() if text_tag else None

        if not text or text == "" or text in seen_reviews:
            continue
            
        seen_reviews.add(text)

        date_tag = review.find("li", class_="review-date")
        date = date_tag.text.strip() if date_tag else None

        rating_tag = review.find("span", class_="ipc-rating-star")
        rating = rating_tag.text.strip() if rating_tag else None

        reviews_data.append({
            "review_id": str(uuid.uuid4()),
            "movie": movie_name,
            "movie_id": movie_id,
            "review": text,
            "rating_10": rating,
            "date": date
        })

        movie_review_count += 1

    print(f"Successfully scraped {movie_review_count} unique reviews for {movie_name}.")

    # Data Cleaning and Saving
    df = pd.DataFrame(reviews_data)
    if not df.empty:
        # Convert ratings to numeric safely
        if "rating_10" in df.columns:
            df["rating_10"] = df["rating_10"].astype(str).str.split("/").str[0]
            df["rating_10"] = pd.to_numeric(df["rating_10"], errors="coerce")

        # Convert date column safely
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            
        df = df.reset_index(drop=True)
        safe_movie_name = movie_name.replace(" ", "_")
        file_path = os.path.join(output_dir, f"{safe_movie_name}.csv")
        
        df.to_csv(file_path, index=False, encoding="utf-8", quoting=1)
        print(f"Saved dataset: {file_path}")
    else:
        print(f"Warning: No valid reviews found for {movie_name}. Skipping file creation.")
        
driver.quit()
print("\nAll tasks completed. Browser closed.")