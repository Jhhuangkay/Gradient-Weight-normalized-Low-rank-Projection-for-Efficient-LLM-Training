#!/bin/bash
# Create or clear the output file
output_file="output.txt"
> $output_file

# Get the last 5 commands
commands=$(history | tail -n 11 | awk '{$1=""; print $0}')

# Execute each command and append its output to the file
while read -r command; do
	    echo "Executing: $command" >> $output_file
	        eval $command >> $output_file 2>&1
		    echo -e "\n" >> $output_file
	    done <<< "$commands"

