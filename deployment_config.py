"""
🚀 DEPLOYMENT CONFIGURATION FOR HACKATHON API
=============================================
HTTPS and production deployment settings
"""

import os
import ssl
from pathlib import Path

class DeploymentConfig:
    """Configuration for production deployment with HTTPS"""

    @classmethod
    def get_ssl_config(cls):
        """Get SSL configuration for HTTPS"""

        # Check for custom SSL certificates
        cert_file = os.getenv("SSL_CERT_FILE", "cert.pem")
        key_file = os.getenv("SSL_KEY_FILE", "key.pem")

        # Check if SSL files exist
        cert_path = Path(cert_file)
        key_path = Path(key_file)

        if cert_path.exists() and key_path.exists():
            return {
                "ssl_keyfile": str(key_path),
                "ssl_certfile": str(cert_path),
                "ssl_version": ssl.PROTOCOL_TLS,
                "ssl_cert_reqs": ssl.CERT_NONE,
                "ssl_ca_certs": None,
                "ssl_ciphers": "TLSv1.2"
            }

        # No SSL files found - will run HTTP (development)
        return None

    @classmethod
    def get_uvicorn_config(cls):
        """Get uvicorn server configuration"""

        # Get port from environment (Render, Heroku, etc.)
        port = int(os.getenv("PORT", 8000))

        # Get host configuration
        host = os.getenv("HOST", "0.0.0.0")

        # Base configuration
        config = {
            "host": host,
            "port": port,
            "reload": False,  # Disable for production
            "log_level": "info",
            "access_log": True,
            "forwarded_allow_ips": "*",  # Allow proxy forwarding
            "proxy_headers": True,  # Enable proxy header support
        }

        # Add SSL config if available
        ssl_config = cls.get_ssl_config()
        if ssl_config:
            config.update(ssl_config)
            print(f"🔒 HTTPS enabled with SSL certificates")
        else:
            print(f"⚠️ Running HTTP (development mode)")
            print(f"🔒 For HTTPS: Set SSL_CERT_FILE and SSL_KEY_FILE environment variables")

        return config

    @classmethod
    def get_deployment_info(cls):
        """Get deployment information"""

        port = int(os.getenv("PORT", 8000))
        host = os.getenv("HOST", "0.0.0.0")
        ssl_config = cls.get_ssl_config()

        protocol = "https" if ssl_config else "http"

        return {
            "protocol": protocol,
            "host": host,
            "port": port,
            "ssl_enabled": ssl_config is not None,
            "public_url": f"{protocol}://{host}:{port}",
            "hackathon_endpoint": f"{protocol}://{host}:{port}/hackrx/run",
            "docs_url": f"{protocol}://{host}:{port}/docs",
            "health_check": f"{protocol}://{host}:{port}/health",
            "auth_info": f"{protocol}://{host}:{port}/api/auth/info"
        }

# Platform-specific deployment configurations
RENDER_CONFIG = {
    "build_command": "pip install -r requirements.txt",
    "start_command": "python api_server.py",
    "environment": "python3",
    "auto_deploy": True,
    "health_check_path": "/health"
}

HEROKU_CONFIG = {
    "web": "python api_server.py",
    "worker": "python monitor.py"
}

RAILWAY_CONFIG = {
    "build": {
        "builder": "NIXPACKS"
    },
    "deploy": {
        "startCommand": "python api_server.py"
    }
}

# Docker configuration
DOCKERFILE_CONTENT = """
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "api_server.py"]
"""
