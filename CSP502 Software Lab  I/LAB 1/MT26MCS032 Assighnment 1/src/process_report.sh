#!/bin/bash

# Count the processes running for the current user
process_count=$(ps -u "$USER" --no-headers | wc -l)

# Save the process count in the report file
echo "Number of processes running for $USER: $process_count" > process_report.txt

# Display the report
cat process_report.txt