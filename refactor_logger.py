import os
import re

def refactor_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    modified = False

    # Check for usages
    has_shared_log = 'logger.log' in content
    has_installer_log = 'logger.log' in content
    has_errors_log = 'logger.log' in content
    has_shared_console = 'logger.console' in content
    has_installer_console = 'logger.console' in content

    if not (has_shared_log or has_installer_log or has_errors_log or has_shared_console or has_installer_console):
        return

    # Add import if needed
    # Check if 'from modules import logger' or 'import modules.logger' exists
    if not re.search(r'from modules import .*logger', content) and not re.search(r'import modules.logger', content):
        # Insert import
        # Try to find standard imports block
        lines = content.split('\n')
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                # find last import
                pass
            if line.startswith('class ') or line.startswith('def '):
                insert_idx = i
                break
        
        # refinement: find last from modules import ...
        last_import_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('from modules import') or line.startswith('import modules'):
                last_import_idx = i
        
        if last_import_idx > 0:
            lines.insert(last_import_idx + 1, 'from modules import logger')
        else:
            # insert at top after __future__ or similar
            for i, line in enumerate(lines):
                if not line.startswith('#') and not line.strip() == '' and not line.startswith('from __future__'):
                    insert_idx = i
                    break
            lines.insert(insert_idx, 'from modules import logger')
        
        content = '\n'.join(lines)
        modified = True

    # Replacements
    if has_shared_log:
        content = content.replace('logger.log', 'logger.log')
        modified = True
    if has_installer_log:
        content = content.replace('logger.log', 'logger.log')
        modified = True
    if has_errors_log:
        content = content.replace('logger.log', 'logger.log')
        modified = True
    if has_shared_console:
        content = content.replace('logger.console', 'logger.console')
        modified = True
    if has_installer_console:
        content = content.replace('logger.console', 'logger.console')
        modified = True
    
    # Specific fix for shared.py and installer.py self-references if any
    # (though I already updated them manually mostly)

    if modified and content != original_content:
        print(f"Modifying {filepath}")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

# Walk directory
root_dir = '/home/vlado/dev/sdnext'
for root, dirs, files in os.walk(root_dir):
    if 'venv' in dirs:
        dirs.remove('venv')
    if '.git' in dirs:
        dirs.remove('.git')
        
    for file in files:
        if file.endswith('.py'):
            refactor_file(os.path.join(root, file))
