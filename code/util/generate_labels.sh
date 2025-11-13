#!/bin/bash

# Generate CSV file with EPB labels
OUTPUT_FILE="../data/epb_labels.csv"

# Create header if file doesn't exist
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "filename,has_epb" > "$OUTPUT_FILE"
    echo "✓ Created new CSV file: $OUTPUT_FILE"
else
    echo "✓ Using existing CSV file: $OUTPUT_FILE"
fi

# Counters
new_entries=0
skipped_entries=0

# Function to check if entry exists
entry_exists() {
    local filename="$1"
    grep -q "^${filename}," "$OUTPUT_FILE"
    return $?
}

# Process has-epb directory
if [ -d "../data/has-epb" ]; then
    echo "Processing has-epb directory..."
    for file in ../data/has-epb/*; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            
            if entry_exists "$filename"; then
                echo "  - Skipped: $filename (already exists)"
                ((skipped_entries++))
            else
                echo "$filename,1" >> "$OUTPUT_FILE"
                echo "  + Added: $filename (has EPB)"
                ((new_entries++))
            fi
        fi
    done
fi

# Process no-epb directory
if [ -d "../data/no-epb" ]; then
    echo "Processing no-epb directory..."
    for file in ../data/no-epb/*; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            
            if entry_exists "$filename"; then
                echo "  - Skipped: $filename (already exists)"
                ((skipped_entries++))
            else
                echo "$filename,0" >> "$OUTPUT_FILE"
                echo "  + Added: $filename (no EPB)"
                ((new_entries++))
            fi
        fi
    done
fi

echo ""
echo "=========================================="
echo "CSV file: $OUTPUT_FILE"
echo "New entries added: $new_entries"
echo "Skipped (duplicates): $skipped_entries"
echo "Total entries: $(tail -n +2 "$OUTPUT_FILE" | wc -l)"
echo "=========================================="
