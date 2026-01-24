param(
  [string]$ModelsFile = "models.txt",
  [string]$DataCsv    = "data.csv",
  [string]$SeqCol     = "sequence",
  [string]$LabelCol   = "positive",
  [int]   $MaxLength  = 256,
  [int]   $Seed       = 42,
  [double]$TestSize   = 0.2
)

$models = Get-Content -Path $ModelsFile

foreach ($m0 in $models) {
  $m = $m0.Trim()
  if (-not $m -or $m.StartsWith('#')) { continue }

  # === batch size rules (你可以按显存再调) ===
  $bs = 8
  if ($m -match '1\.3B') { $bs = 1 }
  elseif ($m -match '600m|600M') { $bs = 2 }
  elseif ($m -match '650M') { $bs = 2 }
  elseif ($m -match 'esm3') { $bs = 2 }
  elseif ($m -match 'ankh') { $bs = 1 }
  elseif ($m -match 'dplm') { $bs = 2 }

  Write-Host "`n=== $m | bs=$bs ==="

  python -u plm_lr_benchmark.py `
    --data_csv  $DataCsv `
    --seq_col   $SeqCol `
    --label_col $LabelCol `
    --model_id  $m `
    --batch_size $bs `
    --max_length $MaxLength `
    --seed $Seed `
    --test_size $TestSize

  if ($LASTEXITCODE -ne 0) {
  Write-Host "FAILED: $m (exit code $LASTEXITCODE) -- continue"
  continue
    }

}
