"""
HL7 FHIR R4 Formatter for RespiSense AI
Generates standard FHIR Observation and Bundle resources compliant with Azure Health Data Services (FHIR Server).
Standards: LOINC, SNOMED CT, HL7 FHIR R4.
"""
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

# Medical Coding System Mapping (LOINC & SNOMED CT)
MEDICAL_CODES = {
    "coughing": {
        "display": "Coughing Episode / Paroxysm",
        "snomed_code": "263731006",
        "snomed_display": "Coughing (finding)",
        "loinc_code": "8687-6",
        "loinc_display": "Coughing [PhenX]",
        "category": "respiratory-biomarker",
        "criticality": "high"
    },
    "breathing": {
        "display": "Respiratory Acoustic Pattern / Wheezing Evaluation",
        "snomed_code": "56018004",
        "snomed_display": "Wheezing / Respiration finding",
        "loinc_code": "9279-1",
        "loinc_display": "Respiratory rate",
        "category": "vital-signs",
        "criticality": "low"
    },
    "snoring": {
        "display": "Snoring & Nocturnal Airway Obstruction Episode",
        "snomed_code": "271600006",
        "snomed_display": "Snoring (finding)",
        "loinc_code": "93832-4",
        "loinc_display": "Sleep disturbance acoustic finding",
        "category": "sleep-biomarker",
        "criticality": "medium"
    },
    "sneezing": {
        "display": "Sneezing / Upper Airway Reflex",
        "snomed_code": "16962002",
        "snomed_display": "Sneezing (finding)",
        "loinc_code": "8688-4",
        "loinc_display": "Sneezing [PhenX]",
        "category": "respiratory-biomarker",
        "criticality": "low"
    },
    "crying_baby": {
        "display": "Infant Acoustic Distress / Cry Episode",
        "snomed_code": "271633008",
        "snomed_display": "Infant crying (finding)",
        "loinc_code": "72144-9",
        "loinc_display": "Infant behavioral distress finding",
        "category": "pediatric-distress",
        "criticality": "high"
    },
    "conversation": {
        "display": "Human Speech / Ambient Vocalization",
        "snomed_code": "286369001",
        "snomed_display": "Speech sound (finding)",
        "loinc_code": "LA11874-7",
        "loinc_display": "Speech present",
        "category": "acoustic-context",
        "criticality": "none"
    },
    "background": {
        "display": "Normal Ambient Room Acoustics / Silence",
        "snomed_code": "162076009",
        "snomed_display": "Normal respiratory sounds (finding)",
        "loinc_code": "LA28669-9",
        "loinc_display": "Normal acoustic environment",
        "category": "environmental",
        "criticality": "none"
    }
}

class FHIRFormatter:
    def __init__(self, patient_id: str = "PATIENT-RESPISENSE-001", device_id: str = "DEVICE-ONNX-EDGE-01"):
        self.patient_id = patient_id
        self.device_id = device_id

    def create_observation(
        self,
        predicted_class: str,
        confidence: float,
        probabilities: Dict[str, float],
        rms_energy: float = 0.0,
        direction_angle: float = 0.0,
        duration_sec: float = 3.0,
        timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Creates a valid HL7 FHIR R4 Observation resource for an acoustic biomarker detection.
        Ready for Azure Health Data Services (FHIR Server) ingestion.
        """
        obs_id = str(uuid.uuid4())
        dt = timestamp or datetime.now(timezone.utc).isoformat()
        code_meta = MEDICAL_CODES.get(predicted_class, MEDICAL_CODES["background"])

        # Interpretation based on criticality & confidence
        if code_meta["criticality"] == "high" and confidence > 50.0:
            interpretation_code = "A" # Abnormal
            interpretation_display = "Abnormal / Clinical Alert"
        elif code_meta["criticality"] == "medium":
            interpretation_code = "W" # Warning
            interpretation_display = "Nocturnal Disturbance"
        else:
            interpretation_code = "N" # Normal
            interpretation_display = "Normal Baseline"

        fhir_observation = {
            "resourceType": "Observation",
            "id": obs_id,
            "meta": {
                "versionId": "1",
                "lastUpdated": dt,
                "profile": [
                    "http://hl7.org/fhir/StructureDefinition/vitalsigns",
                    "https://health.azure.com/fhir/StructureDefinition/respiratory-biomarker"
                ]
            },
            "status": "final",
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                            "code": "vital-signs" if code_meta["category"] == "vital-signs" else "exam",
                            "display": "Vital Signs" if code_meta["category"] == "vital-signs" else "Exam"
                        }
                    ],
                    "text": "Respiratory Acoustic Biomarkers"
                }
            ],
            "code": {
                "coding": [
                    {
                        "system": "http://snomed.info/sct",
                        "code": code_meta["snomed_code"],
                        "display": code_meta["snomed_display"]
                    },
                    {
                        "system": "http://loinc.org",
                        "code": code_meta["loinc_code"],
                        "display": code_meta["loinc_display"]
                    }
                ],
                "text": code_meta["display"]
            },
            "subject": {
                "reference": f"Patient/{self.patient_id}",
                "display": "Monitored Respiratory Patient"
            },
            "device": {
                "reference": f"Device/{self.device_id}",
                "display": "RespiSense AI Edge Sentinel (Microsoft ONNX Runtime)"
            },
            "effectiveDateTime": dt,
            "issued": dt,
            "valueQuantity": {
                "value": round(confidence, 2),
                "unit": "%",
                "system": "http://unitsofmeasure.org",
                "code": "%"
            },
            "interpretation": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                            "code": interpretation_code,
                            "display": interpretation_display
                        }
                    ]
                }
            ],
            "component": [
                {
                    "code": {
                        "coding": [{"system": "http://loinc.org", "code": "LA31756-9", "display": "Acoustic Energy / Intensity"}],
                        "text": "RMS Decibels Proxy"
                    },
                    "valueQuantity": {
                        "value": round(rms_energy, 4),
                        "unit": "RMS",
                        "system": "http://unitsofmeasure.org"
                    }
                },
                {
                    "code": {
                        "coding": [{"system": "http://loinc.org", "code": "LA11875-4", "display": "Event Duration"}],
                        "text": "Audio Window Duration"
                    },
                    "valueQuantity": {
                        "value": duration_sec,
                        "unit": "s",
                        "system": "http://unitsofmeasure.org",
                        "code": "s"
                    }
                },
                {
                    "code": {
                        "coding": [{"system": "http://snomed.info/sct", "code": "272741003", "display": "Laterality / Sound Angle"}],
                        "text": "Bedside Spatial Angle"
                    },
                    "valueQuantity": {
                        "value": round(direction_angle, 1),
                        "unit": "deg",
                        "system": "http://unitsofmeasure.org",
                        "code": "deg"
                    }
                }
            ]
        }
        return fhir_observation

    def create_bundle(self, observations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Packages multiple FHIR Observation resources into an HL7 FHIR R4 Transaction/Collection Bundle
        for batch upload to Azure Health Data Services.
        """
        bundle_id = str(uuid.uuid4())
        entries = []
        for obs in observations:
            entries.append({
                "fullUrl": f"urn:uuid:{obs['id']}",
                "resource": obs,
                "request": {
                    "method": "POST",
                    "url": "Observation"
                }
            })

        return {
            "resourceType": "Bundle",
            "id": bundle_id,
            "type": "transaction",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total": len(observations),
            "entry": entries
        }

    def create_consent_resource(self, patient_id: str = "PATIENT-RESPISENSE-001", policy_code: str = "opt-in-zero-audio-retention") -> Dict[str, Any]:
        """
        Generates an HL7 FHIR R4 Consent resource representing patient opt-in for
        ephemeral bedside acoustic telemetry under zero-cloud-audio retention policies.
        """
        consent_id = f"CONSENT-{str(uuid.uuid4())[:8].upper()}"
        now_iso = datetime.now(timezone.utc).isoformat()
        return {
            "resourceType": "Consent",
            "id": consent_id,
            "status": "active",
            "scope": {
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/consentscope",
                    "code": "patient-privacy",
                    "display": "Privacy Consent"
                }]
            },
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                    "code": "CLINRESRCH",
                    "display": "Clinical Research Telemetry"
                }]
            }],
            "patient": {
                "reference": f"Patient/{patient_id}",
                "display": "Monitored Bedside Patient"
            },
            "dateTime": now_iso,
            "policyRule": {
                "coding": [{
                    "system": "http://nightdoc.ai/fhir/policies",
                    "code": policy_code,
                    "display": "Opt-In Local Edge Processing with Zero Cloud Audio Retention"
                }]
            },
            "provision": {
                "type": "permit",
                "period": { "start": now_iso },
                "purpose": [
                    { "system": "http://terminology.hl7.org/CodeSystem/v3-ActReason", "code": "TREAT" },
                    { "system": "http://terminology.hl7.org/CodeSystem/v3-ActReason", "code": "CLINRESRCH" }
                ],
                "data": [{ "meaning": "related", "reference": { "reference": "Observation" } }]
            }
        }

fhir_formatter = FHIRFormatter()
