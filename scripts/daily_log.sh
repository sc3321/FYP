#!/bin/bash

# directory you want to test
DIR="/home/sc/Projects/FYP"

if [ -d "$DIR" ]; then
    OUT="/home/sc/Projects/FYP/notes/log/record_$(date +%d%m%Y).log"
else
    OUT="$HOME/FYP/notes/log/record_$(date +%d%m%Y).log"
fi

{
    echo "note created at $OUT"
    nvim "$OUT"
}
