#!/bin/bash
# Monitor progress of 1000 image analysis

echo "=== PROGRESS MONITOR FOR 1000 IMAGES ANALYSIS ==="
echo ""
echo "Current Results:"
cat power_map_1000images_results.csv 2>/dev/null || echo "No results yet"
echo ""
echo "---"
echo "Recent Log Output:"
tail -20 run_1000images.log 2>/dev/null || echo "No log yet"
echo ""
echo "---"
echo "Process Status:"
if ps aux | grep -v grep | grep "run_1000images_analysis.sh" > /dev/null; then
    echo "✓ Analysis is RUNNING"
    echo ""
    echo "Estimated completion time:"
    echo "  - Each K value takes ~15-20 minutes for 1000 images"
    echo "  - Total K values: 14"
    echo "  - Estimated total time: 3.5-4.5 hours"
else
    echo "✗ Analysis is NOT running"
fi
echo ""
echo "=== To check progress again, run: ./monitor_progress.sh ==="
