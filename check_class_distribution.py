import pickle
import numpy as np
import os

file_path = 'data/train_data.pkl'

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit(1)

try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    if 'labels' not in data:
        print("Key 'labels' not found in pickle data")
        print(f"Keys found: {data.keys()}")
        exit(1)

    labels = data['labels']
    
    # Convert to numpy array
    labels = np.array(labels)
    
    print(f"Total count: {len(labels)}")
    
    unique, counts = np.unique(labels, return_counts=True)
    
    print("\nClass Distribution:")
    print("-" * 20)
    for cls, count in zip(unique, counts):
        print(f"Class {cls}: {count} ({(count/len(labels))*100:.2f}%)")
    print("-" * 20)

except Exception as e:
    print(f"An error occurred: {e}")
