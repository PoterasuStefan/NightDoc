# RespiSense AI – Health & Research Hackathon Roadmap

> **Proiect:** RespiSense AI – Monitorizarea Biomarkerilor Acustici pentru Cercetare Clinica si Sanatate Respiratorie  
> **Hackathon:** Microsoft (Sectiunea: **Health & Research**)  
> **Stack Tehnic:** Python, PyTorch, Microsoft ONNX Runtime, Azure AI Foundry (Content Understanding), HL7 FHIR, FastAPI, React / Tailwind / Fluent UI.

---

## 1. Contextul Proiectului & Viziunea

### Problema Medicala
Bolile respiratorii cronice (Astm, BPOC, Pneumonie, Fibroza Chistica) si crizele acute se manifesta **mai intai prin anomalii acustice** (accese nocturne de tuse, wheezing / respiratie suieratoare, stridor), cu 24-48 de ore inainte de internarea la urgente. In studiile clinice (Clinical Trials) pentru medicamente noi, cercetatorii nu au acces la date obiective din viata de zi cu zi a pacientilor si se bazeaza pe relatari subiective.

### Solutia: RespiSense AI
Un sistem pasiv neinvaziv (la marginea patului pe telefon/tableta) care monitorizeaza biomarkerii acustici fara a compromite intimitatea pacientului:
* **Edge AI (Microsoft ONNX Runtime):** Detecteaza local tusea, wheezing-ul si anomaliile respiratorii in <10ms, fara a trimite audio pe internet (100% confidentialitate / HIPAA compliant).
* **Cloud Intelligence (Azure AI Foundry):** Cand apare o anomalie, trimite telemetria catre Azure AI Content Understanding pentru a genera un raport clinic structurat.
* **Standard Medical (HL7 FHIR):** Datele sunt formatate nativ ca resurse FHIR pentru integrare in sistemele spitalicesti prin **Azure Health Data Services**.

---

## 2. Ce Este DEJA Construit & Testat (Starea Actuala)

Toate fisierele se afla in folderul `backend/`:
1. **Dataset:** `backend/ESC-50-master/` (contine dataset-ul audio si `meta/esc50.csv`).
2. **Pipeline de Antrenare:** `backend/train_local_model.py`
   - Transforma sunetul in Mel-Spectrograme (`torchaudio.transforms`).
   - Citeste fisierele audio cu `soundfile`.
   - Antreneaza un model convolutional usor (`LightSoundCNN`).
3. **Modele Salvate:**
   - `backend/sound_radar_model.pt` (ponderile native PyTorch).
   - `backend/sound_radar_model.onnx` (formatul de inferenta Microsoft ONNX).
4. **Scripturi de Testare:**
   - `backend/test_onnx.py` (test de inferenta sintetica cu `onnxruntime`).
   - `backend/test_real_audio.py` (inferenta pe fisiere `.wav` reale cu calcul de incredere/probabilitati).

---

## 3. Planul de Implementare Pas cu Pas (De urmat in noul chat)

### Pasul 1: Adaptarea Claselor Modelului pentru Sanatate (Health Dataset Mapping)
* **Obiectiv:** Re-antrenarea rapida a modelului ONNX pe clase relevante medical din ESC-50 si sunete respiratorii:
  * `coughing` (tuse)
  * `breathing` (respiratie normala / wheezing)
  * `crying_baby` (detresa neonatala)
  * `sneezing` (stranut)
  * `speech` / `ambient` (zgomot normal de fond)
* **Timp estimat:** 5 minute (refolosind `train_local_model.py`).

### Pasul 2: Backend-ul FastAPI & Pipeline-ul Hibrid (Edge + Azure)
* **Fisiere de creat:**
  * `backend/main.py`: Server FastAPI cu WebSocket pentru streaming audio bidirectional.
  * `backend/audio_processor.py`: Incarca `sound_radar_model.onnx` cu ONNX Runtime si clasifica segmentele audio in timp real (5-10ms).
  * `backend/azure_foundry.py`: Client pentru **Azure AI Content Understanding** din Foundry care primeste alertele critice si extrage schema clinica.
  * `backend/fhir_formatter.py`: Converteste rezultatele in resurse standard **HL7 FHIR Observation** (gata de trimis in Azure Health Data Services).

### Pasul 3: Frontend-ul Web (Fluent UI / Tailwind)
* **Interfata cu 2 vederi (Toggle simplu):**
  1. **Bedside Patient Sentinel (Modul Pacient):**
     * Ecran minimalist, dark-mode (sa nu deranjeze noaptea).
     * Indicator verde „Privacy Protected – On-Device AI Active”.
     * Afisare in timp real a starii: *Liniste*, *Tuse detectata*, *Respiratie normala*.
     * Foloseste `Screen Wake Lock API` (ecranul nu se stinge) si `Vibration API`.
  2. **Clinical Research Dashboard (Modul Cercetator):**
     * Timeline cu episoadele de tuse nocturna pe ore.
     * Grafic de severitate (Low / Medium / High).
     * Buton „Export to Azure FHIR / EHR”.
     * Rezumat generat de Azure AI pentru raportul clinic.

### Pasul 4: Demo Script & Materiale pentru Prezentare (Pitch)
* Scenariul de 3 minute pe scena in fata juriului Microsoft.
* Raspunsurile pregatite la intrebarile juriului despre:
  * **Confidentialitate:** Modelul ONNX ruleaza pe procesor/NPU local, audio-ul brut este sters instant din memorie.
  * **Scalabilitate & Costuri:** Doar anomaliile clinice ajung in cloud -> reducere de 95% a costurilor.
  * **Ecosistem Microsoft:** ONNX Runtime, Azure AI Foundry, Azure Health Data Services (FHIR).

---

## 4. Prompt de Copiat in Noul Chat

Cand deschizi un chat nou, copiaza si trimite mesajul de mai jos:

```text
Salut! Lucrez la proiectul „RespiSense AI” pentru un hackathon Microsoft (sectiunea Health & Research).
Am deja structura de baza si fisierul PROJECT_ROADMAP.md in radacina proiectului.

Ce avem deja gata in folderul backend/:
1. Dataset-ul audio ESC-50 descarcat in backend/ESC-50-master/.
2. Modelul de clasificare audio antrenat si exportat cu succes in:
   - backend/sound_radar_model.pt (PyTorch)
   - backend/sound_radar_model.onnx (Microsoft ONNX)
3. Scripturile de inferenta functionale cu Microsoft ONNX Runtime (test_onnx.py si test_real_audio.py).

Vreau sa continuam conform planului din PROJECT_ROADMAP.md:
Urmatorul pas este Pasul 2: Sa construim serverul FastAPI (backend/main.py), pipeline-ul de inferenta ONNX in timp real prin WebSockets si integrarea cu Azure AI Foundry / schema HL7 FHIR.
Ghideaza-ma si hai sa construim codul!
```
