import os
import requests
from dotenv import load_dotenv, find_dotenv

class TastytradeConnector:
    def __init__(self):
        dotenv_path = find_dotenv()
        load_dotenv(dotenv_path)  # Load environment variables from .env file
        
        self.username = os.getenv('TASTY_USERNAME')
        self.password = os.getenv('TASTY_PASSWORD')
        if not self.username or not self.password:
            raise ValueError("Missing credentials. Set TASTY_USERNAME and TASTY_PASSWORD in .env file")
        
        self.base_url = 'https://api.tastyworks.com'
        self.session_token = None
        self.create_session()

    def create_session(self):
        url = f'{self.base_url}/sessions'
        payload = {
            'login': self.username,
            'password': self.password
        }
        response = requests.post(
            url, 
            headers={'Content-Type': 'application/json'}, 
            json=payload
        )
        response_data = response.json()
        
        if response.status_code == 201:
            self.session_token = response_data['data']['session-token']
            return self.session_token
        raise Exception(f"Failed to create session: {response.status_code}")

    def get_headers(self):
        if not self.session_token:
            self.create_session()
        return {
            'Authorization': self.session_token,
            'Content-Type': 'application/json'
        }

    def create_order(self, account_number, order):
        url = f'{self.base_url}/accounts/{account_number}/orders'
        headers = self.get_headers()
        response = requests.post(url, headers=headers, json=order)
        
        if response.status_code == 401:
            self.create_session()
            response = requests.post(url, headers=self.get_headers(), json=order)
        
        return response.json() if response.status_code in [200, 201] else None

    def get_account_balance(self, account_number):
        url = f'{self.base_url}/accounts/{account_number}/balances'
        headers = self.get_headers()
        response = requests.get(url, headers=headers)
        
        if response.status_code == 401:
            self.create_session()
            response = requests.get(url, headers=self.get_headers())
        
        return response.json() if response.status_code == 200 else None

if __name__ == "__main__":
    connector = TastytradeConnector()
    account_number = os.getenv('TASTY_ACCOUNT_NUMBER')
    if account_number:
        balance = connector.get_account_balance(account_number)
        print(balance)