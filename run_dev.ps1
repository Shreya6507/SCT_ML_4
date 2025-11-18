<#
run_dev.ps1 - helper to create a venv, install requirements and run the Flask app

Usage (PowerShell):
  .\run_dev.ps1

What it does:
 - creates a local venv at .\.venv if missing
 - installs packages from requirements.txt into the venv
 - runs app.py using the venv's python

This avoids needing to manually activate the venv.
#>

$cwd = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent
Set-Location $cwd

$venv = Join-Path $cwd '.venv'
$python = Join-Path $venv 'Scripts\python.exe'

if (-not (Test-Path $venv)) {
    Write-Host "Creating virtual environment at $venv..."
    python -m venv $venv
}

if (-not (Test-Path $python)) {
    Write-Error "Python in venv not found at $python"
    exit 1
}

Write-Host "Installing requirements into venv (this may take a while)..."
& $python -m pip install --upgrade pip
& $python -m pip install -r requirements.txt

Write-Host "Starting Flask app using venv python..."
Write-Host "To stop the server: Ctrl+C in this window or close it."
& $python app.py
