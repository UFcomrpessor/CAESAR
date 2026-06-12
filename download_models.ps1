$ErrorActionPreference = "Stop"

$ModelDir = "pretrained"
New-Item -ItemType Directory -Frce -Path $ModelDir | Out-Null

$HF_URL = "https://huggingface.co/E53klasky/UFL_CAESAR_MODELS/resolve/main/caesar_v.pt"
$ModelName = "caesar_v.pt"
$OutputPath = "$ModelDir/$ModelName"

Write-Host "Downloading $ModelName from Hugging Face..."
Invoke-WebRequest -Uri $HF_URL -OutFile $OutputPath -UseBasicParsing

if (!(Test-Path $OutputPath) -or (Get-Item $OutputPath).Length -lt 1MB) {
    Write-Host "ERROR: Download failed or file is too small (likely an error page)."
    Write-Host "File size: $((Get-Item $OutputPath -ErrorAction SilentlyContinue).Length) bytes"
    exit 1
}

Write-Host "Downloaded $ModelName successfully ($((Get-Item $OutputPath).Length) bytes)"
Write-Host "All models are in $ModelDir/"
