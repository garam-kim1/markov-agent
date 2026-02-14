#!/bin/bash

# Read stdin into a variable
INPUT=$(cat)

# Use jq to extract tool_name and file_path
TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name')
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path')

# Only proceed if tool is write_file or replace and FILE_PATH is not null
if [[ "$TOOL_NAME" == "write_file" || "$TOOL_NAME" == "replace" ]] && [[ "$FILE_PATH" != "null" ]]; then
    
    # Resolve path relative to GEMINI_PROJECT_DIR if it's not absolute
    PROJECT_DIR="${GEMINI_PROJECT_DIR:-.}"
    FULL_PATH="$FILE_PATH"
    
    if [[ "$FULL_PATH" != /* ]]; then
        FULL_PATH="$PROJECT_DIR/$FILE_PATH"
    fi

    # Check if file exists and ends with .py
    if [[ -f "$FULL_PATH" && "$FULL_PATH" == *.py ]]; then
        # Run ruff format
        if uv run ruff format "$FULL_PATH" > /dev/null 2>&1; then
            echo "Successfully formatted $FULL_PATH using ruff" >&2
        else
            echo "Ruff format failed for $FULL_PATH" >&2
        fi
    fi
fi

# Always return success JSON on stdout
echo '{"continue": true}'
