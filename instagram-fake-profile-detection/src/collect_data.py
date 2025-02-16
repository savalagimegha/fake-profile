import pandas as pd
import random
import os

def collect_data():
    # Ensure the data directory exists
    os.makedirs('data', exist_ok=True)

    # Sample data
    usernames = ['user1', 'user2', 'user3', 'user4', 'user5', 'user6', 'user7', 'user8', 'user9', 'user10']
    bios = [
        'Loving life!', 'Follow me for more!', 'Adventurer', 'Business inquiries only', 'Travel blogger', 
        'Food lover', 'Gamer and streamer', 'Entrepreneur', 'Dreamer', 'Nature enthusiast'
    ]
    
    # Data array to hold profiles
    data = []

    # Populate data with random values and labels
    for i in range(len(usernames)):
        profile = {
            "username": usernames[i],
            "bio": bios[i] if i < len(bios) else "",  # Ensure bios are matched or default to empty
            "followers": random.randint(100, 50000),  # Random followers count
            "following": random.randint(50, 10000),   # Random following count
            "posts": random.randint(1, 500),          # Random number of posts
            "is_fake": random.choice([0, 1])          # Randomly assign 0 (Real) or 1 (Fake)
        }
        data.append(profile)

    # Convert data to DataFrame
    df = pd.DataFrame(data)

    # Save to Excel file
    excel_file_path = 'data/instagram_data.xlsx'
    df.to_excel(excel_file_path, index=False)
    print(f"Sample data collected and saved to '{excel_file_path}'")

if __name__ == '__main__':
    # Run the data collection
    collect_data()
