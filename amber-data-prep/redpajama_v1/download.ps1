$baseUrl = 'https://data.together.xyz/redpajama-data-1T/v1.0.0/'
$urlsFile = Join-Path $PSScriptRoot 'urls.txt'

Get-Content $urlsFile | ForEach-Object {
    $line = $_.Trim()

    if (-not [string]::IsNullOrWhiteSpace($line)) {
        $downloadLocation = $line -replace [regex]::Escape($baseUrl), ''
        $parentDirectory = Split-Path -Parent $downloadLocation

        if (-not [string]::IsNullOrWhiteSpace($parentDirectory)) {
            New-Item -ItemType Directory -Path $parentDirectory -Force | Out-Null
        }

        Invoke-WebRequest -Uri $line -OutFile $downloadLocation
    }
}