param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("a","b","c")]
    [string]$NodeId
)

$configPath = "configs/node_$NodeId.yaml"
Write-Host "Starting node $NodeId with config $configPath"
python -m edge_scheduler.main $configPath
