import os, re

base = r'android/app/src/main/assets'

def read_code(folder):
    path = os.path.join(base, folder, 'code.html')
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

print('Builder helper works!')
