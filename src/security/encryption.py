"""
IB Credential Encryption
========================

Secure encryption/decryption for IB Gateway credentials using Fernet (AES-256).
"""

import os
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


def get_encryption_key() -> bytes:
    """
    Get or generate encryption key.
    
    Uses IB_ENCRYPTION_KEY env var if set, otherwise generates from a secret.
    """
    key = os.getenv("IB_ENCRYPTION_KEY")
    
    if key:
        return key.encode()
    
    # Fallback: derive key from a secret phrase
    # In production, always use IB_ENCRYPTION_KEY env var
    secret = os.getenv("SECRET_KEY", "ib-trading-platform-default-key")
    salt = b"ib-trading-salt"
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(secret.encode()))
    return key


def generate_new_key() -> str:
    """Generate a new Fernet key for IB_ENCRYPTION_KEY."""
    return Fernet.generate_key().decode()


class CredentialEncryptor:
    """Encrypt and decrypt IB credentials."""
    
    def __init__(self):
        self.key = get_encryption_key()
        self.fernet = Fernet(self.key)
    
    def encrypt(self, username: str, password: str) -> str:
        """
        Encrypt IB username and password.
        
        Args:
            username: IB username
            password: IB password
            
        Returns:
            Base64-encoded encrypted string
        """
        data = f"{username}:{password}".encode()
        encrypted = self.fernet.encrypt(data)
        return encrypted.decode()
    
    def decrypt(self, encrypted_data: str) -> tuple:
        """
        Decrypt IB credentials.
        
        Args:
            encrypted_data: Encrypted string from encrypt()
            
        Returns:
            Tuple of (username, password)
        """
        decrypted = self.fernet.decrypt(encrypted_data.encode())
        data = decrypted.decode()
        username, password = data.split(":", 1)
        return username, password


# Singleton instance
_encryptor = None

def get_encryptor() -> CredentialEncryptor:
    """Get singleton encryptor instance."""
    global _encryptor
    if _encryptor is None:
        _encryptor = CredentialEncryptor()
    return _encryptor


def encrypt_credentials(username: str, password: str) -> str:
    """Convenience function to encrypt credentials."""
    return get_encryptor().encrypt(username, password)


def decrypt_credentials(encrypted: str) -> tuple:
    """Convenience function to decrypt credentials."""
    return get_encryptor().decrypt(encrypted)


if __name__ == "__main__":
    print("IB Credential Encryption Utility")
    print("=" * 40)
    
    # Generate a new key for .env
    new_key = generate_new_key()
    print(f"\nNew encryption key for .env:")
    print(f"IB_ENCRYPTION_KEY={new_key}")
    
    # Test encryption
    print("\nTesting encryption...")
    test_user = "test_user"
    test_pass = "test_password123"
    
    encrypted = encrypt_credentials(test_user, test_pass)
    print(f"Encrypted: {encrypted[:50]}...")
    
    user, pwd = decrypt_credentials(encrypted)
    print(f"Decrypted: {user} / {'*' * len(pwd)}")
    
    assert user == test_user and pwd == test_pass
    print("\n✅ Encryption working correctly!")
