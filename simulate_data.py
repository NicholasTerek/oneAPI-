import random
import time

def simulate_wastewater_data():
    """Simulate wastewater data collection."""
    return {
        "bacteria_count": random.randint(100, 1000),
        "virus_rna_level": round(random.uniform(0.1, 10.0), 2),
        "antibiotic_resistance": round(random.uniform(0.0, 1.0), 2)
    }

if __name__ == "__main__":
    while True:
        data = simulate_wastewater_data()
        print(f"Simulated Data: {data}")
        time.sleep(1)
