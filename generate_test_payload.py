"""
Generate a valid test payload for the fraud detection API
"""
import pickle
import json
import random

# Load the feature names the model expects
with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

print(f"Model expects {len(feature_names)} features")
print(f"Features: {feature_names[:20]}...\n")

# Create a sample transaction with all required features
transaction = {}

for feature in feature_names:
    if feature == 'time_step':
        transaction[feature] = 5  # Time period 1-49
    elif feature in ['f2', 'f3']:  # Amount and fee
        transaction[feature] = random.uniform(0.1, 10.0)
    elif feature in ['f4', 'f5']:  # Inputs/outputs
        transaction[feature] = random.randint(1, 5)
    else:
        # Random values for other features
        transaction[feature] = random.uniform(-1.0, 1.0)

# Save to file
with open('test_payload.json', 'w') as f:
    json.dump(transaction, f, indent=2)

print("✅ Generated test_payload.json")
print("\nSample payload (first 10 fields):")
sample = {k: transaction[k] for k in list(transaction.keys())[:10]}
print(json.dumps(sample, indent=2))

print("\n📋 Copy the content of test_payload.json to use in Postman")
