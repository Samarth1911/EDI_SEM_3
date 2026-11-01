# train_hawkes_ensembles.ps1 -- complete the spread-Hawkes arm from 3 to 5 training seeds so 5-member ensembles exist.
# NOT RUN by its author. Commands are the Stage 3 handoff's own Hawkes commands with --seeds 5 (same hyperparameters, same tag).
# Run from EDI\validation with the venv active:    .\train_hawkes_ensembles.ps1
# Resumable: belief caches and finished (arm, seed) pairs are skipped on rerun. KEEP THE MACHINE AWAKE (a suspended run was 25x slower).
# Rough time from the Gaussian numbers in the handoff (Hawkes not measured): train cache about 42 min per seed (85 s / 1000 worlds,
# 30000 worlds) x 2 seeds, plus about 10-11 min per arm-seed x 4 arms x 2 new seeds = about 2.5-3 h in total.
# DISCLOSE in the write-up: seeds 3 and 4 were added after the 3-seed Hawkes results had been seen (optional stopping risk); the
# Hawkes arm stays exploratory and the earlier 3-seed numbers are superseded, not mixed with the 5-seed ones.
$ErrorActionPreference = "Stop"
$arms = "oracle", "raw_proxy", "filt_full", "filt_mean_only"

function Invoke-Step($cmd) {
    Write-Host "`n>> $cmd" -ForegroundColor Cyan
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) { throw "failed: $cmd" }
}

# 1. Belief caches for the two NEW training seeds (the val/test caches already exist from the 3-seed run).
foreach ($s in 3, 4) {
    Invoke-Step "python build_belief_cache.py --filter imm --obs-kind spread_hawkes --what train --train-seed $s --episodes 30000"
}

# 2. Training. Seeds 0-2 are already in outputs\stage3c\results_base1_hawkes.jsonl and are skipped; seeds 3-4 are trained.
Invoke-Step "python run_stage3c_multiseed.py --arms oracle,raw_proxy,filt_full,filt_mean_only --seeds 5 --eval-episodes 200 --target-sync 2000 --n-val 300 --episodes 30000 --obs-kind spread_hawkes --use-cache --tag _hawkes"

# 3. Verify that all 4 arms x 5 seeds have a validation-selected checkpoint.
$missing = @()
foreach ($a in $arms) {
    foreach ($k in 0..4) {
        $p = "outputs\stage3c\ckpt\${a}_base1_hawkes_s${k}_valsel.npz"
        if (-not (Test-Path $p)) { $missing += $p }
    }
}
if ($missing.Count -gt 0) {
    Write-Host "`nMISSING checkpoints:" -ForegroundColor Red
    $missing
    exit 1
}
Write-Host "`nall 20 valsel checkpoints present (4 arms x 5 seeds)" -ForegroundColor Green
Write-Host "next: python compare_arms.py --tags _hawkes --out-name hawkes_report   (now n = 5 per arm)"