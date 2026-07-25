# locustfile.py
# ─────────────────────────────────────────────────────────────
# Load test for the cancer mutation classifier /predict endpoint.
# Run with: locust -f locustfile.py --host http://localhost:8000
# Then open http://localhost:8089 to configure and start the test.
# ─────────────────────────────────────────────────────────────

from locust import HttpUser, task, between
import random

# A few varied realistic payloads so we're not hammering the exact
# same cached-friendly input every time.
SAMPLE_REQUESTS = [
    {
        "gene": "BRCA1",
        "variation": "R1699Q",
        "clinical_text": "The BRCA1 gene plays a critical role in DNA repair via homologous recombination and is frequently mutated in hereditary breast and ovarian cancer."
    },
    {
        "gene": "TP53",
        "variation": "R175H",
        "clinical_text": "TP53 is a tumor suppressor gene that regulates the cell cycle and prevents genomic mutations. R175H is a well-known hotspot mutation."
    },
    {
        "gene": "EGFR",
        "variation": "L858R",
        "clinical_text": "EGFR L858R is a common activating mutation found in non-small cell lung cancer, leading to constitutive kinase activity."
    },
    {
        "gene": "KRAS",
        "variation": "G12D",
        "clinical_text": "KRAS G12D is one of the most frequent oncogenic mutations in pancreatic and colorectal cancers, locking the protein in an active state."
    },
]


class PredictUser(HttpUser):
    # Wait 1-3 seconds between requests per simulated user —
    # mimics realistic traffic instead of hammering nonstop.
    wait_time = between(1, 3)

    @task
    def predict(self):
        payload = random.choice(SAMPLE_REQUESTS)
        self.client.post("/predict", json=payload)

    @task(1)  # runs less often than predict
    def health_check(self):
        self.client.get("/health")