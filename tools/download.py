import os
import urllib.request
import json

def download_file(url, file_name):
    if not os.path.exists(file_name):
        try:
            urllib.request.urlretrieve(url, file_name)
            print(f"Downloaded to {file_name}")
            return True
        except Exception as e:
            print(f"Download failed: {e}")
            return False
    return True  # Already exists


# def download_and_load_json_file(file_path, url):
#     if not os.path.exists(file_path):
#         with urllib.request.urlopen(url) as response:
#             text_data = response.read().decode("utf-8")
#         with open(file_path, "w", encoding="utf-8") as file:
#             file.write(text_data)

#     with open(file_path, "r", encoding="utf-8") as file:
#         data = json.load(file)
#     return data


def download_and_load_json_file(url, file_name):
    """
    Downloads and loads JSON content from a URL if not already present.
    Returns parsed JSON object or None on failure.
    """
    try:
        # Download phase
        if not download_file(url, file_name):
            return None

        # Load JSON Phase
        with open(file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"Loaded JSON from {file_name}")
            return data

    except FileNotFoundError:
        print(f"File not found after download: {file_name}")
    except json.JSONDecodeError as je:
        print(f"Invalid JSON in {file_name}: {je}")
    except Exception as e:
        print(f"Unexpected error loading JSON: {e}")

    return None