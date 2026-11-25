# Keeps exactly one live bot.py running. Run in its own terminal:
#     powershell -ExecutionPolicy Bypass -File .\watchdog.ps1
#
# "Is the bot alive" is decided by the OS lock on bot.lock itself, not by searching
# process command lines: the lock is the definition of "a live submitter exists", and test
# runs (which use a different LOCK_FILE) are correctly ignored.
#
# Safe against double submission: if this ever races with a manual start, the second
# bot.py cannot take the lock and exits. It never stops a healthy bot.

$ErrorActionPreference = 'Continue'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $here
$logFile  = [System.IO.Path]::Combine($here, 'logs', 'watchdog.log')
$lockFile = [System.IO.Path]::Combine($here, 'bot.lock')
$interval = 20

function Write-Watchdog([string]$msg) {
    $line = "{0} {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg
    Write-Output $line
    try { Add-Content -LiteralPath $logFile -Value $line -Encoding utf8 } catch {}
}

# Resolve python once, up front, and say which one -- a silent PATH problem is exactly
# what broke the first version of this script.
$py = $null
try { $py = (Get-Command python -ErrorAction Stop).Source } catch {}
if (-not $py) {
    $fallback = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python311\python.exe'
    if (Test-Path -LiteralPath $fallback) { $py = $fallback }
}
if (-not $py) {
    Write-Watchdog "FATAL: cannot find python.exe - watchdog cannot restart anything"
    exit 1
}

function Test-BotAlive {
    # Try to take the same byte-range lock the bot holds. If we can, nobody holds it.
    try {
        $fs = [System.IO.File]::Open($lockFile, 'OpenOrCreate', 'ReadWrite', 'ReadWrite')
    } catch {
        return $true          # cannot even open it: assume alive, never double-start
    }
    try {
        $fs.Lock(0, 1)
        $fs.Unlock(0, 1)
        return $false
    } catch {
        return $true
    } finally {
        $fs.Close()
    }
}

function Start-Bot {
    $env:ALLOW_LIVE = '1'
    foreach ($k in 'ONLY_GAME', 'DRY_RUN', 'LOCK_FILE') { Remove-Item "Env:$k" -ErrorAction SilentlyContinue }
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName         = $py
    $psi.Arguments        = 'bot.py'
    $psi.WorkingDirectory = $here          # .NET takes the path literally (no wildcards)
    $psi.UseShellExecute  = $true          # own console window; inherits our environment
    $psi.WindowStyle      = 'Minimized'
    $p = [System.Diagnostics.Process]::Start($psi)
    return $p.Id
}

if ($args -contains '-selftest') {
    Write-Output "python : $py"
    Write-Output "lock   : $lockFile"
    Write-Output ("alive  : {0}" -f (Test-BotAlive))
    exit 0
}

Write-Watchdog "watchdog started (pid $PID), python=$py, checking every ${interval}s"
$restarts = 0

while ($true) {
    if (-not (Test-BotAlive)) {
        # Never start inside the 60 s submission window: wait for it to pass.
        $base = Get-Date -Year 2026 -Month 11 -Day 21 -Hour 14 -Minute 0 -Second 0
        $sinceRound = [math]::Floor(((((Get-Date) - $base).TotalSeconds) % 800))
        if ($sinceRound -lt 70) {
            Write-Watchdog "bot is DOWN but a round is in progress (${sinceRound}s in); waiting"
        } else {
            $restarts++
            Write-Watchdog "bot is DOWN - restarting (restart #$restarts)"
            try {
                $newPid = Start-Bot
                Write-Watchdog "restart issued, new pid $newPid"
            } catch {
                Write-Watchdog "restart FAILED: $($_.Exception.Message)"
            }
        }
    }
    Start-Sleep -Seconds $interval
}
