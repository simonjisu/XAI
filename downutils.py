import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argparser",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path Settings
    parser.add_argument("-to", "--destination", type=str, required=True, help="destination")
    parser.add_argument("-id", "--file_id", typ=str, required=True, help="file id")
    args = parser.parse_args()

    file_id = args.file_id
    destination = args.destination
    download_file_from_google_drive(file_id, destination)