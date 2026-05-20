#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${PORT:-8522}"
URL="http://127.0.0.1:${PORT}/slides.html"
LOG="${DIR}/.slides-server.log"
PIDFILE="${DIR}/.slides-server.pid"

if ! (echo > "/dev/tcp/127.0.0.1/${PORT}") >/dev/null 2>&1; then
  (cd "$DIR" && nohup python3 -m http.server "$PORT" --bind 127.0.0.1 > "$LOG" 2>&1 & echo $! > "$PIDFILE")
  sleep 0.8
fi

if grep -qi microsoft /proc/version 2>/dev/null; then
  powershell.exe -NoProfile -Command "Start-Process '$URL'" >/dev/null 2>&1 || {
    echo "Open failed. Copy this URL into Windows browser: $URL" >&2
    exit 1
  }
elif command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$URL"
elif command -v open >/dev/null 2>&1; then
  open "$URL"
else
  echo "$URL"
fi

echo "Opened: $URL"
echo "Server log: $LOG"
