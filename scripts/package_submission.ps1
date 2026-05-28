param(
    [Parameter(Mandatory=$true)][string]$FirstName,
    [Parameter(Mandatory=$true)][string]$LastName,
    [string]$CheckpointPath = "outputs\checkpoints_multitask_late_fusion\best.pth"
)

$zipName = "${FirstName}_${LastName}_GalaxEye.zip"
$staging = "outputs\submission"

New-Item -ItemType Directory -Force -Path $staging | Out-Null

Copy-Item -Force $CheckpointPath "$staging\best.pth"
Copy-Item -Force reports\technical_report.pdf "$staging\technical_report.pdf"
Copy-Item -Force reports\time_resource_log.txt "$staging\time_resource_log.txt"

if (Test-Path $zipName) {
    Remove-Item $zipName
}

Compress-Archive -Path "$staging\*" -DestinationPath $zipName
Write-Host "Created $zipName"
