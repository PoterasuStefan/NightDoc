import os
import sys
import httpx

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_DIR = os.path.join(BASE_DIR, "Additional_Manual_Training")

CATEGORIES = {
    "tuse_seaca_dry": {
        "pattern": "_cough-shallow.wav",
        "limit": 15,
        "description": "Tuse Seaca / Iritativa (Dry Cough - Shallow)"
    },
    "tuse_productiva_heavy": {
        "pattern": "_cough-heavy.wav",
        "limit": 15,
        "description": "Tuse Productiva / Grea (Wet / Heavy Paroxysmal Cough)"
    },
    "respiratie_profunda_wheezing": {
        "pattern": "_breathing-deep.wav",
        "limit": 15,
        "description": "Respiratie Profunda & Wheezing (Deep Respiration)"
    },
    "respiratie_superficiala_detresa": {
        "pattern": "_breathing-shallow.wav",
        "limit": 15,
        "description": "Respiratie Rapida / Detresa (Shallow Tachypnea)"
    }
}

def main():
    print("==================================================================")
    print("  [*] Descarcam mostre clinice audio COUGHVID / Coswara (~15/categorie)")
    print("==================================================================")
    
    api_url = "https://huggingface.co/api/datasets/marceltomas/covid-cough-detection/tree/main/data/wavs16k"
    
    try:
        print("[1/3] Preluam lista de fisiere audio din dataset-ul medical...")
        res = httpx.get(api_url, timeout=15.0, follow_redirects=True)
        if res.status_code != 200:
            print(f"Eroare API: {res.status_code}")
            return
        
        all_files = res.json()
        print(f"[2/3] Gasite {len(all_files)} fisiere audio disponibile.")
        
        base_raw_url = "https://huggingface.co/datasets/marceltomas/covid-cough-detection/resolve/main/"
        
        # Descarcam pentru fiecare categorie
        for cat_name, cat_info in CATEGORIES.items():
            cat_dir = os.path.join(TARGET_DIR, cat_name)
            os.makedirs(cat_dir, exist_ok=True)
            
            pattern = cat_info["pattern"]
            limit = cat_info["limit"]
            matched_files = [f["path"] for f in all_files if f["path"].endswith(pattern)][:limit]
            
            print(f"\n---> Categoria: {cat_info['description']} ({len(matched_files)} fisiere)")
            
            count = 0
            for file_path in matched_files:
                filename = os.path.basename(file_path)
                dest_path = os.path.join(cat_dir, filename)
                
                if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
                    count += 1
                    continue
                
                download_url = base_raw_url + file_path
                try:
                    r = httpx.get(download_url, timeout=20.0, follow_redirects=True)
                    if r.status_code == 200:
                        with open(dest_path, "wb") as f:
                            f.write(r.content)
                        count += 1
                        print(f"   [+] Descarcat ({count}/{limit}): {filename} ({len(r.content)} bytes)")
                    else:
                        print(f"   [!] HTTP {r.status_code} la {filename}")
                except Exception as err:
                    print(f"   [!] Eroare descarcare {filename}: {err}")
                    
            print(f"   [OK] Total {count} fisiere salvate in: backend/Additional_Manual_Training/{cat_name}/")
            
        print("\n==================================================================")
        print("  [*] TOATE MOSTRELE AU FOST DESCARCATE CU SUCCES!")
        print("==================================================================")
        
    except Exception as e:
        print(f"Eroare generala: {e}")

if __name__ == "__main__":
    main()
