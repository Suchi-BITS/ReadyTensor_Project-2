import requests, re
from typing import List, Optional, Dict
import json

class User:
    def __init__(self, base_url: str = "http://localhost:8080/api/v1", token: str|None = None):
        self.base_url = base_url.rstrip('/')
        self.token = token

    def set_token(self, token: str) -> None:
        self.token = token

    # Generate a root token. This can only be done once for the server.
    def generate_root_token(self) -> str:
        response = requests.post(f"{self.base_url}/root/token")
        return response.text

    # Create a new user using root credentials and return the user token.
    def create_user(self, username: str, root_token: str) -> str:
        headers = {"Authorization": root_token}
        data = {"username": username}
        response = requests.post(
            f"{self.base_url}/users",
            headers=headers,
            json=data
        )
        return response.text

    # Delete a user using root credentials. It deletes the user, his indexes and all associated tokens.
    def delete_user(self, username: str) -> None:
        # Check if user is root user
        if self.token.split(":")[0] != "root":
            raise Exception("Only root user can delete other users")
        if username == "root":
            raise Exception("Cannot delete root user")
        headers = {"Authorization": self.token}
        response = requests.delete(
            f"{self.base_url}/users/{username}",
            headers=headers
        )
        response.raise_for_status()
        return response.text

    # It return the user as not active and deletes all his tokens
    def deactivate_user(self, username: str) -> None:
        headers = {"Authorization": self.token}
        if username == "root":
            raise Exception("Cannot deactivate root user")
        if self.token.split(":")[0] != "root":
            raise Exception("Only root user can deactivate other users")
        response = requests.post(
            f"{self.base_url}/users/{username}/deactivate",
            headers=headers
        )
        response.raise_for_status()

    # Generate a new token for the authenticated user
    def generate_token(self, name: str) -> str:
        if self.token is None:
            raise Exception("User token not set. Please set the user token using the set_token method.")
        # Name should not have space or special characters. Only alphanum and _ allowed
        if not re.match(r'^[a-zA-Z0-9_]+$', name):
            raise Exception("Token name should only contain alphanumeric characters and _")
        headers = {"Authorization": self.token}
        data = {"name": name}
        response = requests.post(
            f"{self.base_url}/tokens",
            headers=headers,
            json=data
        )
        print(response.text)
        response.raise_for_status()
        return response.text

    # List all tokens for the authenticated user.
    def list_tokens(self) -> List[Dict]:
        if self.token is None:
            raise Exception("User token not set. Please set the user token using the set_token method.")
        headers = {"Authorization": self.token}
        response = requests.get(
            f"{self.base_url}/tokens",
            headers=headers
        )
        response.raise_for_status()
        return response.json()["tokens"]

    # Delete a specific token.
    def delete_token(self, token_name: str) -> None:
        if self.token is None:
            raise Exception("User token not set. Please set the user token using the set_token method.")
        headers = {"Authorization": self.token}
        response = requests.delete(
            f"{self.base_url}/tokens/{token_name}",
            headers=headers
        )
        response.raise_for_status()
    # Get detailed information about a user (requires root or self)
    def get_user_info(self, username: str) -> Dict:
        if self.token is None:
            raise Exception("User token not set. Please set the user token using the set_token method.")
        headers = {"Authorization": self.token}
        response = requests.get(
            f"{self.base_url}/users/{username}/info",
            headers=headers
        )
        response.raise_for_status()
        return response.json()

    # Get the type of a user (Free, Starter, or Pro)
    def get_user_type(self, username: str) -> str:
        if self.token is None:
            raise Exception("User token not set. Please set the user token using the set_token method.")
        headers = {"Authorization": self.token}
        response = requests.get(
            f"{self.base_url}/users/{username}/type",
            headers=headers
        )
        response.raise_for_status()
        return response.json()["user_type"]

    # Set the type of a user (requires root)
    def set_user_type(self, username: str, user_type: str) -> None:
        if self.token is None:
            raise Exception("User token not set. Please set the user token using the set_token method.")
        if self.token.split(":")[0] != "root":
            raise Exception("Only root user can set user types")
        if user_type not in ["Free", "Starter", "Pro"]:
            raise Exception("User type must be one of: Free, Starter, Pro")
        
        headers = {"Authorization": self.token}
        data = {"user_type": user_type}
        response = requests.put(
            f"{self.base_url}/users/{username}/type",
            headers=headers,
            json=data
        )
        response.raise_for_status()

    # Get a list of all indices across all users (requires root)
    def get_all_indices(self) -> List[Dict]:
        if self.token is None:
            raise Exception("User token not set. Please set the user token using the set_token method.")
        if self.token.split(":")[0] != "root":
            raise Exception("Only root user can view all indices")
        
        headers = {"Authorization": self.token}
        response = requests.get(
            f"{self.base_url}/index/all",
            headers=headers
        )
        response.raise_for_status()
        return response.json()["indices"]

