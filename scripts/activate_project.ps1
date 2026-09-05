$ProjectRoot = 'D:\CycAMP-FusionDemo'
$CacheRoot = Join-Path $ProjectRoot '.cache'

$env:CONDA_PKGS_DIRS = Join-Path $CacheRoot 'conda'
$env:HF_HOME = Join-Path $CacheRoot 'huggingface'
$env:HUGGINGFACE_HUB_CACHE = Join-Path $env:HF_HOME 'hub'
$env:TRANSFORMERS_CACHE = Join-Path $env:HF_HOME 'transformers'
$env:TORCH_HOME = Join-Path $CacheRoot 'torch'
$env:PIP_CACHE_DIR = Join-Path $CacheRoot 'pip'
$env:TEMP = Join-Path $CacheRoot 'tmp'
$env:TMP = Join-Path $CacheRoot 'tmp'
$env:PYTHONPATH = Join-Path $ProjectRoot 'src'

$RequiredDirectories = @(
    $env:CONDA_PKGS_DIRS,
    $env:HF_HOME,
    $env:HUGGINGFACE_HUB_CACHE,
    $env:TRANSFORMERS_CACHE,
    $env:TORCH_HOME,
    $env:PIP_CACHE_DIR,
    $env:TEMP
)

foreach ($Directory in $RequiredDirectories) {
    New-Item -ItemType Directory -Force -Path $Directory | Out-Null
}

Write-Host "CycAMP-FusionDemo cache variables now point to $CacheRoot"

