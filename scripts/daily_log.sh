#!/bin/bash

OUT="/home/sc/Projects/FYP/notes/log/record_$(date +%d%m%Y).log"

{
    echo "note created in $OUT/.."
    nvim $OUT
}


