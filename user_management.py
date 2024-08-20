# user_management.py
fake_users_db = {}

def get_user_wallet(user_id):
    """
    Fetch the user's wallet details from the in-memory database or actual data store.
    """
    user = fake_users_db.get(user_id)
    if user and "wallet" in user:
        return user["wallet"]
    else:
        raise ValueError(f"No wallet found for user_id: {user_id}")
