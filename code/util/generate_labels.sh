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

# Create all-images directory if it doesn't exist
mkdir -p "../data/all-images"

# Move images from has-epb to all-images
moved_has_epb=0
if [ -d "../data/has-epb" ]; then
    moved_has_epb=$(find ../data/has-epb -type f | wc -l)
    if [ $moved_has_epb -gt 0 ]; then
        if mv ../data/has-epb/* "../data/all-images/" 2>/dev/null; then
            echo "✓ Moved $moved_has_epb images from has-epb to all-images"
        else
            echo "✗ Failed to move images from has-epb"
            moved_has_epb=0
        fi
    fi
fi

# Move images from no-epb to all-images
moved_no_epb=0
if [ -d "../data/no-epb" ]; then
    moved_no_epb=$(find ../data/no-epb -type f | wc -l)
    if [ $moved_no_epb -gt 0 ]; then
        if mv ../data/no-epb/* "../data/all-images/" 2>/dev/null; then
            echo "✓ Moved $moved_no_epb images from no-epb to all-images"
        else
            echo "✗ Failed to move images from no-epb"
            moved_no_epb=0
        fi
    fi
fi

echo ""
echo "=========================================="
echo "Total images moved: $((moved_has_epb + moved_no_epb))"
echo "=========================================="
