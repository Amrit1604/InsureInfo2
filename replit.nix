# Replit configuration for free hosting
[nix]
channel = "stable-22_11"

[deployment]
run = "python api_server.py"
deploymentTarget = "cloudrun"

[[ports]]
localPort = 8000
externalPort = 80
