#!/bin/bash
# Monitor training progress from the log file
LOG="logs/training.log"
echo "=== Training Monitor ==="
echo "Last 10 training iterations:"
grep "iteration.*lm_loss" "$LOG" | tail -10
echo ""
echo "=== GPU Memory ==="
grep "after.*iterations memory" "$LOG" | tail -1
echo ""
echo "=== Errors ==="
grep -i "error\|traceback\|OOM\|nan iterations:   [^0]" "$LOG" | tail -5
echo ""
echo "=== Disk ==="
df -h /
