"""
Create RDS Database
===================
Since AWS Query Editor is unavailable, run this script to 
remotely create the 'ib_trading' database on your existing RDS instance.

Usage:
1. Update DB_PASSWORD below.
2. Run: python create_rds_db.py
"""

import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# ================= Configuration =================
DB_HOST = "travelwise-marketplace-db.curmg864eafo.us-east-1.rds.amazonaws.com"
#DB_USER = "postgres"  
DB_USER = "erichuang2005" 
         # Default master username
DB_PASSWORD = "Ya2039349"  # <--- REPLACE THIS WITH YOUR REAL PASSWORD
# =================================================

def create_database():
    print(f"Connecting to {DB_HOST}...")
    
    try:
        # Connect to default 'postgres' db to execute CREATE DATABASE
        conn = psycopg2.connect(
            dbname="marketplace",
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port="5432",
            sslmode="require"
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = 'ib_trading'")
        exists = cursor.fetchone()
        
        if not exists:
            print("Creating database 'ib_trading'...")
            cursor.execute("CREATE DATABASE ib_trading")
            print("✅ SUCCESS: Database 'ib_trading' created successfully!")
        else:
            print("⚠️ Database 'ib_trading' already exists.")
            
    except psycopg2.OperationalError as e:
        print(f"❌ Connection Failed: {e}")
        print("Tip: Check your password and ensure your IP is allowed in the RDS Security Group.")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()

if __name__ == "__main__":
    if DB_PASSWORD == "YOUR_PASSWORD":
        print("❌ Please edit this file and set DB_PASSWORD first.")
        sys.exit(1)
        
    create_database()
