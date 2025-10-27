#!/bin/bash

# Generate CSV file with EPB labels
OUTPUT_FILE="epb_labels.csv"

# Create header
echo "filename,has_epb" > "$OUTPUT_FILE"

# Process has-epb directory
if [ -d "All_Sky_images/has-epb" ]; then
    for file in All_Sky_images/has-epb/*; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            echo "$filename,1" >> "$OUTPUT_FILE"
        fi
    done
fi

# Process no-epb directory
if [ -d "All_Sky_images/no-epb" ]; then
    for file in All_Sky_images/no-epb/*; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            echo "$filename,0" >> "$OUTPUT_FILE"
        fi
    done
fi

echo "CSV file generated: $OUTPUT_FILE"
echo "Total entries: $(tail -n +2 "$OUTPUT_FILE" | wc -l)"
