import os

def sanitize_env():
    if not os.path.exists('.env'):
        print(".env not found")
        return

    with open('.env', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    sanitized_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            sanitized_lines.append(line) # Keep comments and empty lines as is (but line itself is stripped of \r\n by .strip() then kept or not?)
            # Wait, if I want to preserve structure:
            if not stripped:
                sanitized_lines.append('\n')
            else:
                sanitized_lines.append(stripped + '\n')
            continue
        
        if '=' in stripped:
            key, value = stripped.split('=', 1)
            sanitized_lines.append(f"{key.strip()}={value.strip()}\n")
        else:
            sanitized_lines.append(stripped + '\n')

    with open('.env', 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(sanitized_lines)
    
    print("Successfully sanitized .env file.")

if __name__ == "__main__":
    sanitize_env()
