# 一键启动 / One-shot launcher (Windows PowerShell 5.1+)
#
#   .\scripts\start.ps1
#
# 创建 Python 虚拟环境、安装前后端依赖，然后同时拉起 FastAPI (:8000)
# 与 Vite (:5173)。Ctrl+C 会一并停止两个服务（含子进程树）。
#
# 常用参数：
#   -BackendOnly            只启动后端
#   -SkipInstall            跳过 pip / pnpm install
#   -BackendPort 8001       换端口
#   -FrontendPort 5174
#   -Python "py -3.12"      指定 Python 解释器
#
# 注意：本文件必须保存为 **UTF-8 with BOM**。Windows PowerShell 5.1 对无 BOM
# 文件按 ANSI 代码页（简中为 GBK）解码，会把下面的中文变成乱码并破坏引号
# 配对，导致「表达式或语句中包含意外的标记」这类看似莫名其妙的解析错误。

[CmdletBinding()]
param(
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 5173,
    [switch]$SkipInstall,
    [switch]$BackendOnly,
    [string]$Python = ""
)

$ErrorActionPreference = 'Stop'

# 传统代码页控制台下，中文 Write-Host 会乱码；切到 UTF-8。
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $OutputEncoding = [System.Text.Encoding]::UTF8
} catch {
    # 非交互宿主（输出被重定向）——仅影响观感，继续执行。
}

$root = Split-Path -Parent $PSScriptRoot
$backend = Join-Path $root 'backend'
$frontend = Join-Path $root 'frontend'
$venvPython = Join-Path $backend '.venv\Scripts\python.exe'

function Write-Step($message) {
    Write-Host ""
    Write-Host "==> $message" -ForegroundColor Cyan
}

function Write-Warn($message) {
    Write-Host "!!  $message" -ForegroundColor Yellow
}

function Write-Err($message) {
    Write-Host "XX  $message" -ForegroundColor Red
}

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

# TCP 连通性探测。用裸 socket 而不是 Invoke-WebRequest，避免 PS 5.1 在
# 无 IE 引擎时对 -UseBasicParsing 的额外要求。
function Test-TcpPort {
    param([string]$ComputerName, [int]$Port, [int]$TimeoutMs = 800)
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $iar = $client.BeginConnect($ComputerName, $Port, $null, $null)
        if (-not $iar.AsyncWaitHandle.WaitOne($TimeoutMs, $false)) { return $false }
        $client.EndConnect($iar)
        return $true
    } catch {
        return $false
    } finally {
        $client.Close()
    }
}

function Wait-TcpPort {
    param([string]$ComputerName, [int]$Port, [int]$Attempts = 40, [int]$IntervalMs = 500)
    for ($i = 0; $i -lt $Attempts; $i++) {
        if (Test-TcpPort -ComputerName $ComputerName -Port $Port) { return $true }
        Start-Sleep -Milliseconds $IntervalMs
    }
    return $false
}

function Resolve-Python {
    if ($Python -ne '') { return $Python }
    # 优先 3.13：科学计算与可选后端的 wheel 覆盖最完整。
    foreach ($candidate in @('py -3.13', 'py -3.12', 'py -3.11', 'python')) {
        $parts = $candidate.Split(' ')
        if (Get-Command $parts[0] -ErrorAction SilentlyContinue) { return $candidate }
    }
    throw "找不到 Python。请安装 Python 3.11–3.13 后重试，或用 -Python 指定解释器。"
}

# 解析出一个**真正的可执行** pnpm 启动器。
#
# 关键点：`Start-Process -FilePath 'pnpm'` 在 Windows 上会失败并报
# "%1 is not a valid Win32 application"。原因是 npm 全局目录下同时存在
# 三个同名文件：`pnpm`（Unix bash 脚本，无扩展名）、`pnpm.cmd`、`pnpm.ps1`。
# Start-Process 把 'pnpm' 直接交给 CreateProcess，命中的是那个 bash 脚本，
# Windows 无法执行它。因此必须显式解析到 .cmd / .exe，再用 cmd.exe 启动。
function Resolve-NodeLauncher {
    $candidates = New-Object System.Collections.Generic.List[string]

    foreach ($name in @('pnpm', 'npm')) {
        $commands = @(Get-Command $name -All -ErrorAction SilentlyContinue)
        foreach ($command in $commands) {
            $source = $command.Source
            if ([string]::IsNullOrEmpty($source)) { continue }
            # 同名但无扩展名的 Unix shim 必须排除。
            $candidates.Add($source)
            $candidates.Add("$source.cmd")
            $candidates.Add("$source.exe")
        }
    }

    $candidates.Add((Join-Path $env:APPDATA 'npm\pnpm.cmd'))
    $candidates.Add((Join-Path $env:APPDATA 'npm\npm.cmd'))
    $candidates.Add((Join-Path $env:LOCALAPPDATA 'pnpm\pnpm.exe'))

    foreach ($candidate in $candidates) {
        if ([string]::IsNullOrWhiteSpace($candidate)) { continue }
        if ($candidate -notmatch '\.(cmd|exe|bat)$') { continue }
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
    }
    return $null
}

# 杀掉整棵进程树。cmd.exe -> node.exe(vite) 的父子关系意味着只杀父进程
# 会留下占着端口的孤儿 node 进程。
function Stop-ProcessTree {
    param([int]$ProcessId)
    if (-not $ProcessId) { return }
    & taskkill.exe /F /T /PID $ProcessId 2>&1 | Out-Null
}

# ---------------------------------------------------------------------------
# 预检
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "泥石流坡度滚落物理动态分析系统 — 启动中" -ForegroundColor Green
Write-Host "项目根目录: $root"

if (-not (Test-Path -LiteralPath $backend)) { throw "找不到后端目录: $backend" }
if (-not $BackendOnly -and -not (Test-Path -LiteralPath $frontend)) {
    throw "找不到前端目录: $frontend"
}

if (Test-TcpPort -ComputerName '127.0.0.1' -Port $BackendPort) {
    Write-Warn "端口 $BackendPort 已被占用。若那是上一次残留的后端进程，请先结束它，" `
        + "或用 -BackendPort 换一个端口。"
}
if (-not $BackendOnly -and (Test-TcpPort -ComputerName '127.0.0.1' -Port $FrontendPort)) {
    Write-Warn "端口 $FrontendPort 已被占用。可能是上次残留的 Vite，" `
        + "或用 -FrontendPort 换一个端口。"
}

# ---------------------------------------------------------------------------
# 后端
# ---------------------------------------------------------------------------
if (-not (Test-Path -LiteralPath $venvPython)) {
    $python = Resolve-Python
    Write-Step "创建 Python 虚拟环境 ($python)"
    $parts = $python.Split(' ')
    & $parts[0] $parts[1..($parts.Length - 1)] -m venv (Join-Path $backend '.venv')
    if (-not (Test-Path -LiteralPath $venvPython)) { throw "虚拟环境创建失败：$venvPython" }
}

if (-not $SkipInstall) {
    Write-Step "安装后端依赖"
    & $venvPython -m pip install --upgrade pip --quiet
    & $venvPython -m pip install -r (Join-Path $backend 'requirements.txt') --quiet
    if ($LASTEXITCODE -ne 0) { throw "后端依赖安装失败（pip 退出码 $LASTEXITCODE）" }
}

Write-Step "启动后端 http://127.0.0.1:$BackendPort"
# Declared up front so the finally block can always reference them, even if a
# later Start-Process throws.
$backendJob = $null
$frontendJob = $null
$backendJob = Start-Process -FilePath $venvPython `
    -ArgumentList @('-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', "$BackendPort") `
    -WorkingDirectory $backend -PassThru -NoNewWindow

# ---------------------------------------------------------------------------
# 前端
# ---------------------------------------------------------------------------
if (-not $BackendOnly) {
    $launcher = Resolve-NodeLauncher
    if ($null -eq $launcher) {
        Write-Err "找不到可用的 pnpm / npm 启动器。"
        Write-Host "    请安装 Node.js (>= 20) 后执行:  npm install -g pnpm" -ForegroundColor DarkGray
        Stop-ProcessTree -ProcessId $backendJob.Id
        exit 1
    }
    Write-Host "    使用包管理器: $launcher" -ForegroundColor DarkGray

    if (-not (Test-Path -LiteralPath (Join-Path $frontend 'node_modules'))) {
        Write-Step "安装前端依赖"
        Push-Location $frontend
        try {
            & $launcher install
            if ($LASTEXITCODE -ne 0) { throw "前端依赖安装失败（退出码 $LASTEXITCODE）" }
        } finally {
            Pop-Location
        }
    }

    Write-Step "启动前端 http://127.0.0.1:$FrontendPort"
    # 用 cmd.exe 启动 .cmd，CreateProcess 无法直接执行批处理。
    $frontendJob = Start-Process -FilePath 'cmd.exe' `
        -ArgumentList @('/c', $launcher, 'dev', '--port', "$FrontendPort") `
        -WorkingDirectory $frontend -PassThru -NoNewWindow
}

# ---------------------------------------------------------------------------
# 就绪确认
# ---------------------------------------------------------------------------
$backendReady = Wait-TcpPort -ComputerName '127.0.0.1' -Port $BackendPort -Attempts 60
if ($backendReady) {
    Write-Host "    后端已就绪" -ForegroundColor DarkGray
} else {
    Write-Warn "后端在 30 秒内没有监听 $BackendPort，请查看上方 uvicorn 输出。"
}

if (-not $BackendOnly) {
    $frontendReady = Wait-TcpPort -ComputerName '127.0.0.1' -Port $FrontendPort -Attempts 60
    if ($frontendReady) {
        Write-Host "    前端已就绪" -ForegroundColor DarkGray
    } else {
        Write-Warn "前端在 30 秒内没有监听 $FrontendPort，请查看上方 Vite 输出。"
    }
}

Write-Host ""
if ($backendReady -and ($BackendOnly -or $frontendReady)) {
    Write-Host "就绪：" -ForegroundColor Green
} else {
    Write-Host "服务未能全部就绪：" -ForegroundColor Yellow
}
if (-not $BackendOnly) {
    Write-Host "  前端      http://127.0.0.1:$FrontendPort"
    Write-Host "  （请用上面的前端地址访问界面；直接打开 :$BackendPort 只会看到后端 JSON）"
}
Write-Host "  后端 API  http://127.0.0.1:$BackendPort"
Write-Host "  接口文档  http://127.0.0.1:$BackendPort/docs"
Write-Host ""
Write-Host "按 Ctrl+C 停止。" -ForegroundColor DarkGray

# ---------------------------------------------------------------------------
# 守护循环 + 清理
# ---------------------------------------------------------------------------
try {
    while ($true) {
        Start-Sleep -Seconds 1
        if ($backendJob.HasExited) {
            Write-Host "后端进程已退出（代码 $($backendJob.ExitCode)）" -ForegroundColor Yellow
            break
        }
        if ($frontendJob -and $frontendJob.HasExited) {
            Write-Host "前端进程已退出（代码 $($frontendJob.ExitCode)）" -ForegroundColor Yellow
            break
        }
    }
} finally {
    Write-Step "停止服务"
    foreach ($job in @($backendJob, $frontendJob)) {
        if ($job -and -not $job.HasExited) {
            Stop-ProcessTree -ProcessId $job.Id
        }
    }
    Start-Sleep -Milliseconds 500
}
