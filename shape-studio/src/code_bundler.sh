#!/bin/bash

# Define the output file
OUTPUT_FILE="codebase_bundle.txt"

# Clear the output file if it exists
> "$OUTPUT_FILE"

# Find files recursively
# We limit to .svelte, .js, and .css as requested
find . -type f \( -name "*.svelte" -o -name "*.js" -o -name "*.css" \) -not -path "*/node_modules/*" -not -path "*/.*" | while read -r file; do
    
    # Remove the './' prefix for a cleaner relative path
    relative_path="${file#./}"
    
    echo "Processing: $relative_path"

    # Write the separator and metadata
    echo "--- START OF FILE: $relative_path ---" >> "$OUTPUT_FILE"
    
    # Append the actual file content
    cat "$file" >> "$OUTPUT_FILE"
    
    # Add a newline and a closing separator for clarity
    echo -e "\n--- END OF FILE: $relative_path ---\n" >> "$OUTPUT_FILE"

done

echo "Done! All code has been bundled into $OUTPUT_FILE"