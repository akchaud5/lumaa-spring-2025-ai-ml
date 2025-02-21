from kaggle.api.kaggle_api_extended import KaggleApi
import os

def download_dataset():
    """Download the TMDB movie dataset from Kaggle"""
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Initialize and authenticate Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    # Download dataset
    print("Downloading TMDB dataset...")
    api.dataset_download_files(
        'tmdb/tmdb-movie-metadata',
        path='data',
        unzip=True
    )
    print("Dataset downloaded successfully to ./data directory")

if __name__ == "__main__":
    try:
        download_dataset()
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nMake sure you have:")
        print("1. Created a Kaggle account")
        print("2. Generated an API token from Settings page")
        print("3. Placed kaggle.json in ~/.kaggle/")
        print("4. Set correct permissions with: chmod 600 ~/.kaggle/kaggle.json")