#!/bin/bash
OUT="env_$(date +%Y%m%d_%H%M%S).log"
{
  echo "=== SYSTEM INFO ==="
  uname -a
  lsb_release -a 2>/dev/null || cat /etc/os-release
  echo
  echo "=== CPU INFO ==="
  lscpu
  echo
} > "$OUT"
echo "Environment captured in scripts/$OUT"

