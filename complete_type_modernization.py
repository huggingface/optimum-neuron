#!/usr/bin/env python3
"""Complete type hint modernization script"""

import os
import re
import subprocess

def modernize_complex_type_hints(content):
    """Modernize complex type hints using more sophisticated patterns"""
    
    # Handle Union with complex nested types
    # This pattern handles A | B where A and B can contain brackets
    def replace_union(match):
        inner = match.group(1)
        # Split on commas that are not inside brackets
        parts = []
        bracket_count = 0
        current_part = ""
        
        for char in inner:
            if char in '([{':
                bracket_count += 1
            elif char in ')]}':
                bracket_count -= 1
            elif char == ',' and bracket_count == 0:
                parts.append(current_part.strip())
                current_part = ""
                continue
            current_part += char
        
        if current_part.strip():
            parts.append(current_part.strip())
        
        if len(parts) == 2:
            return f"{parts[0]} | {parts[1]}"
        else:
            # For more than 2 parts, join with |
            return " | ".join(parts)
    
    content = re.sub(r'Union\[([^]]+)\]', replace_union, content)
    
    # Handle simple cases that might have been missed
    content = re.sub(r'Optional\[([^]]+)\]', r'\1 | None', content)
    content = re.sub(r'List\[([^]]+)\]', r'list[\1]', content)
    content = re.sub(r'Dict\[([^]]+)\]', r'dict[\1]', content)
    content = re.sub(r'Tuple\[([^]]+)\]', r'tuple[\1]', content)
    content = re.sub(r'Set\[([^]]+)\]', r'set[\1]', content)
    
    return content

def process_file(file_path):
    """Process a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        content = modernize_complex_type_hints(content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f'Error processing {file_path}: {e}')
        return False

def find_files_with_old_types():
    """Find all Python files with old type hints"""
    result = subprocess.run(
        ['find', '.', '-name', '*.py', '-type', 'f'],
        capture_output=True, text=True
    )
    files = result.stdout.strip().split('\n')
    
    files_with_old_types = []
    for file_path in files:
        if not file_path or not os.path.exists(file_path):
            continue
            
        try:
            result = subprocess.run(
                ['grep', '-l', '-E', '(Union\\[|Optional\\[|List\\[|Dict\\[|Tuple\\[|Set\\[)', file_path],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                files_with_old_types.append(file_path)
        except:
            continue
    
    return files_with_old_types

def main():
    print("=== Finding files with old type hints ===")
    files_to_process = find_files_with_old_types()
    print(f"Found {len(files_to_process)} files to process")
    
    updated_files = []
    
    for file_path in files_to_process:
        if process_file(file_path):
            updated_files.append(file_path)
            print(f"✓ Updated: {file_path}")
    
    if updated_files:
        print(f"\n=== Committing {len(updated_files)} updated files ===")
        for file_path in updated_files:
            subprocess.run(['git', 'add', file_path], check=True)
        
        commit_msg = "refactor(types): complete type hint modernization across entire codebase"
        subprocess.run(['git', 'commit', '-m', commit_msg], check=True)
        print(f"✓ Committed: {commit_msg}")
    else:
        print("No files needed updating")

if __name__ == "__main__":
    main()