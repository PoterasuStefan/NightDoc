import os, re

base_dir = r'android/app/src/main/assets'

def read_file(rel_path):
    p = os.path.join(base_dir, rel_path)
    with open(p, 'r', encoding='utf-8') as f:
        return f.read()

def extract_body_content(html):
    m = re.search(r'<main[^>]*>(.*?)</main>', html, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ''

print('Script loaded successfully')
