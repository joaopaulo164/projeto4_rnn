# scripts/setup.ps1 - Environment Setup
Write-Host "Setting up Python virtual environment and dependencies..."
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
Write-Host "Setup completed."
