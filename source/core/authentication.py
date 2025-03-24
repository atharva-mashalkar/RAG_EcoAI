import yaml
import os

CREDENTIALS_FILE = 'config/credentials.yaml'

def load_credentials():
    if not os.path.exists(CREDENTIALS_FILE):
        return {}
    with open(CREDENTIALS_FILE, 'r') as file:
        return yaml.safe_load(file) or {}


def save_credentials(credentials):
    with open(CREDENTIALS_FILE, 'w') as file:
        yaml.safe_dump(credentials, file)


def signup(firstname, lastname, username, password):
    credentials = load_credentials()
    if username in credentials:
        return False, "Username already exists."
    credentials[username] = {
        'firstname': firstname,
        'lastname': lastname,
        'password': password
    }
    save_credentials(credentials)
    return True, "Signup successful."


def login(username, password):
    credentials = load_credentials()
    if username in credentials and credentials[username]['password'] == password:
        return True, "Login successful."
    return False, "Invalid username or password."
