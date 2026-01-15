# Cloud Deployment Guide 🚀

This document contains everything you need to deploy the IB Trading System to the cloud (AWS EC2) using Git.

## 1. Prerequisites

### Local Machine (Windows)
- [x] **Git** installed.
- [x] **GitHub/GitLab Repository** created.

### Cloud Server (AWS EC2)
- [x] **Ubuntu 22.04 LTS** instance running.
- [x] **Ports Open** (Security Group):
    - 22 (SSH)
    - 80/443 (HTTP/HTTPS)
    - 5000 (Dashboard)
    - 5900 (VNC - Optional for debugging)

---

## 2. One-Time Setup

### A. Link Local Project to Git
Run these commands locally once to connect your project to GitHub/GitLab:

```powershell
# Initialize git
git init

# Add your repository URL (Replace with your actual URL)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
```

### B. Setup Server
SSH into your server. I have provided a helper script for this:

```powershell
# Usage: scripts\ssh_login.bat <YOUR_EC2_IP>
.\scripts\ssh_login.bat 1.2.3.4
```

Once connected, clone the repository:

```bash
# 1. Update system
sudo apt update && sudo apt install -y git


# 2. Clone repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git ib-program-trading

# 3. Enter directory
cd ib-program-trading

# 4. Run setup script (Installs Docker, Nginx, etc.)
chmod +x deploy.sh
./deploy.sh
```

---

## 3. One-Click Deployment

I have created a single script that handles **everything** in one command:

```powershell
.\scripts\deploy_full.bat <YOUR_EC2_IP>
```
*Example: `.\scripts\deploy_full.bat 54.123.45.67`*

**What it does automatically:**
1. Commits all local changes to Git
2. Pushes to your remote repository
3. SSHs into EC2 using your PEM key
4. Pulls the latest code on the server
5. Restarts Docker containers

**First-time setup:**
Edit `scripts\deploy_full.bat` and set:
- `EC2_IP` = Your server's IP address
- `REPO_URL` = Your GitHub repository URL

---

## 4. Post-Deployment Verification


1.  **Check Logs:**
    ```bash
    docker-compose logs -f trading-system
    ```
    *Look for "System Started" and "RL Agents loaded".*

2.  **Access Dashboard:**
    Open `http://YOUR_SERVER_IP:5000` in your browser.

3.  **VNC Access (Optional):**
    Connect to `YOUR_SERVER_IP:5900` with VNC Viewer to see the TWS Gateway interface.

---

## 5. Troubleshooting

-   **"Permission denied" during git push:** Check your git credentials or SSH keys.
-   **"Docker not found":** Ensure `deploy.sh` ran successfully on the server.
-   **RL Agents not loading:** Ensure `checkpoints/` folder was committed and pushed (check `.gitignore`).
