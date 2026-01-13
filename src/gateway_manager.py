"""
Gateway Manager
===============

Dynamically manages IB Gateway Docker containers for each user.
Handles spawning, port allocation, health checks, and shutdown.
"""

import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    print("Warning: docker package not installed. Gateway manager will be limited.")

from models.multi_tenant import MultiTenantDB
from security.encryption import decrypt_credentials

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============= Configuration =============

BASE_PORT = 4001
MAX_GATEWAYS = 10
GATEWAY_IMAGE = "ghcr.io/gnzsnz/ib-gateway:latest"


class GatewayManager:
    """Manage IB Gateway containers for multiple users."""
    
    def __init__(self):
        self.db = MultiTenantDB()
        self.docker_client = docker.from_env() if DOCKER_AVAILABLE else None
        self.active_gateways: Dict[str, dict] = {}  # user_id -> gateway info
        
        logger.info("Gateway Manager initialized")
    
    def get_available_port(self) -> int:
        """Find next available port."""
        used_ports = set()
        
        # Get ports from database
        for account in self.db.get_all_active_accounts():
            if account.gateway_port:
                used_ports.add(account.gateway_port)
        
        # Find first available
        for port in range(BASE_PORT, BASE_PORT + MAX_GATEWAYS):
            if port not in used_ports:
                return port
        
        raise RuntimeError("No available gateway ports")
    
    def spawn_gateway(self, user_id: str) -> Optional[dict]:
        """
        Spawn IB Gateway container for a user.
        
        Returns:
            Gateway info dict or None if failed
        """
        if not DOCKER_AVAILABLE:
            logger.error("Docker not available")
            return None
        
        # Get IB credentials
        account = self.db.get_ib_account(user_id)
        if not account:
            logger.error(f"No IB account for user {user_id}")
            return None
        
        # Decrypt credentials
        try:
            ib_user, ib_pass = decrypt_credentials(account.credentials_encrypted)
        except Exception as e:
            logger.error(f"Failed to decrypt credentials: {e}")
            return None
        
        # Get available port
        port = self.get_available_port()
        container_name = f"ib-gateway-{user_id[:8]}"
        
        try:
            # Check if already running
            try:
                existing = self.docker_client.containers.get(container_name)
                if existing.status == "running":
                    logger.info(f"Gateway already running for {user_id}")
                    return {
                        "container_id": existing.id,
                        "port": port,
                        "status": "running"
                    }
                else:
                    existing.remove()
            except docker.errors.NotFound:
                pass
            
            # Spawn new container
            logger.info(f"Spawning gateway for user {user_id} on port {port}")
            
            container = self.docker_client.containers.run(
                GATEWAY_IMAGE,
                detach=True,
                name=container_name,
                ports={"4001/tcp": port, "5900/tcp": None},
                environment={
                    "TWS_USERID": ib_user,
                    "TWS_PASSWORD": ib_pass,
                    "TRADING_MODE": account.trading_mode or "paper",
                    "IB_API_PORT": "4001"
                },
                restart_policy={"Name": "unless-stopped"}
            )
            
            # Update database
            self.db.update_gateway_status(user_id, port, connected=True)
            
            gateway_info = {
                "container_id": container.id,
                "container_name": container_name,
                "port": port,
                "status": "starting",
                "trading_mode": account.trading_mode
            }
            
            self.active_gateways[user_id] = gateway_info
            
            logger.info(f"Gateway spawned: {container_name} on port {port}")
            return gateway_info
            
        except Exception as e:
            logger.error(f"Failed to spawn gateway: {e}")
            return None
    
    def stop_gateway(self, user_id: str) -> bool:
        """Stop gateway for a user."""
        if not DOCKER_AVAILABLE:
            return False
        
        container_name = f"ib-gateway-{user_id[:8]}"
        
        try:
            container = self.docker_client.containers.get(container_name)
            container.stop(timeout=10)
            container.remove()
            
            # Update database
            self.db.update_gateway_status(user_id, 0, connected=False)
            
            # Remove from active
            if user_id in self.active_gateways:
                del self.active_gateways[user_id]
            
            logger.info(f"Gateway stopped: {container_name}")
            return True
            
        except docker.errors.NotFound:
            return True
        except Exception as e:
            logger.error(f"Failed to stop gateway: {e}")
            return False
    
    def get_gateway_status(self, user_id: str) -> dict:
        """Get status of user's gateway."""
        account = self.db.get_ib_account(user_id)
        
        if not account:
            return {"status": "not_linked"}
        
        if not DOCKER_AVAILABLE:
            return {
                "status": "unknown",
                "port": account.gateway_port,
                "docker_available": False
            }
        
        container_name = f"ib-gateway-{user_id[:8]}"
        
        try:
            container = self.docker_client.containers.get(container_name)
            return {
                "status": container.status,
                "port": account.gateway_port,
                "trading_mode": account.trading_mode,
                "last_connected": account.last_connected_at.isoformat() if account.last_connected_at else None
            }
        except docker.errors.NotFound:
            return {
                "status": "stopped",
                "port": account.gateway_port,
                "trading_mode": account.trading_mode
            }
    
    def list_all_gateways(self) -> list:
        """List all running gateways."""
        if not DOCKER_AVAILABLE:
            return []
        
        gateways = []
        
        try:
            containers = self.docker_client.containers.list(
                filters={"name": "ib-gateway-"}
            )
            
            for container in containers:
                ports = container.ports.get("4001/tcp", [])
                port = int(ports[0]["HostPort"]) if ports else None
                
                gateways.append({
                    "name": container.name,
                    "status": container.status,
                    "port": port,
                    "id": container.short_id
                })
        except Exception as e:
            logger.error(f"Error listing gateways: {e}")
        
        return gateways
    
    def health_check_all(self) -> dict:
        """Health check all gateways."""
        gateways = self.list_all_gateways()
        
        healthy = sum(1 for g in gateways if g["status"] == "running")
        
        return {
            "total": len(gateways),
            "healthy": healthy,
            "gateways": gateways
        }
    
    def spawn_all_user_gateways(self):
        """Spawn gateways for all users with linked accounts."""
        accounts = self.db.get_all_active_accounts()
        
        logger.info(f"Spawning gateways for {len(accounts)} users")
        
        for account in accounts:
            try:
                self.spawn_gateway(str(account.user_id))
                time.sleep(5)  # Stagger startup
            except Exception as e:
                logger.error(f"Failed to spawn gateway for {account.user_id}: {e}")


# ============= CLI =============

if __name__ == "__main__":
    print("=" * 60)
    print("Gateway Manager")
    print("=" * 60)
    
    if not DOCKER_AVAILABLE:
        print("\n⚠️  Docker package not installed. Install with: pip install docker")
        sys.exit(1)
    
    manager = GatewayManager()
    
    # List gateways
    print("\nActive Gateways:")
    gateways = manager.list_all_gateways()
    if gateways:
        for g in gateways:
            print(f"  {g['name']}: {g['status']} (port {g['port']})")
    else:
        print("  No gateways running")
    
    # Health check
    health = manager.health_check_all()
    print(f"\nHealth: {health['healthy']}/{health['total']} healthy")
