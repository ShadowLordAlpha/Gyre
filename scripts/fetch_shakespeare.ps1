# Downloads Karpathy tiny Shakespeare (~1 MiB continuous char-LM text).
# HuggingFace flwrlabs/shakespeare is LEAF federated CSV, not this format.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $root) { $root = Get-Location }
$dir = Join-Path $root "data"
New-Item -ItemType Directory -Force -Path $dir | Out-Null
$out = Join-Path $dir "shakespeare.txt"
$url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
Write-Host "Downloading $url"
Invoke-WebRequest -Uri $url -OutFile $out -UseBasicParsing
Write-Host "Wrote $out ($((Get-Item $out).Length) bytes)"
