#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-sync_end}
RUNS=${2:-20}

if [[ "$MODE" == "sync_end" ]]; then
    BIN="out/sync_end"
    OUTDIR="runs/syncEnd"
elif [[ "$MODE" == "sync_every" ]]; then
    BIN="out/sync_every"
    OUTDIR="runs/syncEvery"
else
    echo "Usage: ./run_sync_experiment.sh [sync_end|sync_every] [runs]"
    exit 1
fi

mkdir -p "$OUTDIR"

echo "Running $MODE for $RUNS trials"
echo "Binary: $BIN"
echo "Output: $OUTDIR"

durfile="$OUTDIR/all_sync_durations.txt"
waitfile="$OUTDIR/all_wait_durations.txt"

rm -f "$durfile" "$waitfile"

for i in $(seq 1 "$RUNS"); do

    tag=$(printf "%03d" "$i")
    rundir="$OUTDIR/run_$tag"
    mkdir -p "$rundir"

    echo "Run $tag"

    strace -ff -ttt -T -s 0 \
        -yy \
        -e trace=futex,poll,ppoll,epoll_wait,ioctl,nanosleep,clock_nanosleep \
        -o "$rundir/strace" \
        taskset -c 2 "$BIN" \
        2> "$rundir/markers.log"

    awk '
    /MARKER: startTime/ {
        t=$1; gsub(/[\[\]]/,"",t); start=t
    }
    /MARKER: endTime/ {
        t=$1; gsub(/[\[\]]/,"",t); end=t
    }
    END{
        if(start!="" && end!=""){
            dur=end-start
            print start, end, dur
        }
    }' "$rundir/markers.log" > "$rundir/sync_window.txt"

    dur=$(awk '{print $3}' "$rundir/sync_window.txt" || true)
    if [[ -n "${dur:-}" ]]; then
        echo "$dur" >> "$durfile"
    fi

    for f in "$rundir"/strace.*; do
        awk '
        /FUTEX_WAIT/ {
            if(match($0,/<([0-9.]+)>/,a)){
                print a[1]
            }
        }' "$f" >> "$waitfile"
    done

    start=$(awk '{print $1}' "$rundir/sync_window.txt" || true)
    end=$(awk '{print $2}' "$rundir/sync_window.txt" || true)

    if [[ -n "${start:-}" && -n "${end:-}" ]]; then

        pad=0.002

        for f in "$rundir"/strace.*; do
            awk -v s="$start" -v e="$end" -v p="$pad" '
            $1+0 >= (s-p) && $1+0 <= (e+p) {
                print FILENAME ": " $0
            }' "$f"
        done > "$rundir/sync_window_syscalls.txt"
    fi

done

echo "Computing aggregate statistics"

awk '
{a[NR]=$1}
END{
    n=NR
    if(n==0){exit}
    asort(a)
    p50=a[int(n*0.50)]
    p90=a[int(n*0.90)]
    p99=a[int(n*0.99)]
    printf "sync_duration_stats\n"
    printf "n=%d\n",n
    printf "p50=%.6f\n",p50
    printf "p90=%.6f\n",p90
    printf "p99=%.6f\n",p99
    printf "max=%.6f\n",a[n]
}' "$durfile" > "$OUTDIR/sync_duration_stats.txt"

awk '
{a[NR]=$1}
END{
    n=NR
    if(n==0){exit}
    asort(a)
    p50=a[int(n*0.50)]
    p90=a[int(n*0.90)]
    p99=a[int(n*0.99)]
    printf "wait_duration_stats\n"
    printf "n=%d\n",n
    printf "p50=%.6f\n",p50
    printf "p90=%.6f\n",p90
    printf "p99=%.6f\n",p99
    printf "max=%.6f\n",a[n]
}' "$waitfile" > "$OUTDIR/wait_duration_stats.txt"

echo "Finished."
echo "Results stored in $OUTDIR"
