import glob

def fix():
    for file in glob.glob("**/*.py", recursive=True):
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Replace the incorrectly escaped quotes """ with standard quotes """
        if r'"""' in content:
            content = content.replace(r'"""', '"""')
            with open(file, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Fixed {file}")

if __name__ == "__main__":
    fix()
