#!/bin/bash

# File counters
txt_count=0
c_count=0
java_count=0
sh_count=0

# Recursively search all files, including hidden files
while IFS= read -r file
do
    # Get file extension
    extension="${file##*.}"

    # Count files according to their extension
    case "$extension" in
        txt)
            ((txt_count++))
            ;;
        c)
            ((c_count++))
            ;;
        java)
            ((java_count++))
            ;;
        sh)
            ((sh_count++))
            ;;
    esac

done < <(find . -type f)

# Print formatted table
echo "================================"
echo "       FILE COUNT REPORT"
echo "================================"
printf "%-10s %s\n" "Extension" "Count"
echo "--------------------------------"

# Sort output by file count in descending order
{
    printf "%d .txt\n" "$txt_count"
    printf "%d .c\n" "$c_count"
    printf "%d .java\n" "$java_count"
    printf "%d .sh\n" "$sh_count"
} | sort -rn

echo "================================"