import time
import csv
import os
import numpy as np
from mnist_loader import LocalMNISTLoader
from quantum_cortex import QuantumCortex
from fourier_optics import FourierOptics

# Remove 'time' from arguments
def log_experiment(train_acc, test_acc, duration, config, notes):
    filename = "quantum_validation_log.csv"
    file_exists = os.path.isfile(filename)
    
    # Generate timestamp here instead
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Timestamp", "Train_Acc", "Test_Acc", "Duration", "Config", "Notes"])
        writer.writerow([
            timestamp,
            f"{train_acc:.2f}%",
            f"{test_acc:.2f}%",
            f"{duration:.1f}s",
            str(config),
            notes
        ])
    print(f"\n[Log] Validation results saved to {filename}")

def run_validation():
    # data_path = r"C:\Users\Maxwell Wilson\OneDrive\Documents\Quantum_snn_benchmark\mnist_data"
    data_path = r"./mnist_data"
    loader = LocalMNISTLoader(data_path)
    
    # --- THE 90% WINNING CONFIG ---
    NEURONS_PER_CLASS = 5
    TRAIN_SAMPLES = 60000
    
    physics_config = {
        'learning_rate':    0.09,
        'phase_flexibility': 0.1,
        'lateral_strength':  0.16,
        'input_threshold':   0.7,  # The High Gate
        'kerr_constant':     0.2,
        'system_energy':     40.0  # High Gain
    }
    
    try:
        print(f"--- Quantum Cortex Validation Run ---")
        print(f"Physics: {physics_config}")
        
        # 1. LOAD TRAINING DATA
        print("-> Loading Training Data...")
        train_images = loader.load_images('train-images.idx3-ubyte')
        train_labels = loader.load_labels('train-labels.idx1-ubyte')
        
        # Shuffle Training Data
        indices = np.arange(len(train_images))
        np.random.shuffle(indices)
        train_images = train_images[indices]
        train_labels = train_labels[indices]

        # 2. LOAD TEST DATA (The Final Exam)
        print("-> Loading Test Data...")
        try:
            test_images = loader.load_images('t10k-images.idx3-ubyte')
            test_labels = loader.load_labels('t10k-labels.idx1-ubyte')
        except:
            print("!! TEST DATA NOT FOUND !!")
            print("Please ensure 't10k-images.idx3-ubyte' and 't10k-labels.idx1-ubyte' are in the folder.")
            return

        # 3. INITIALIZE TRINITY
        print("-> Initializing Quantum Trinity...")
        optics = FourierOptics(shape=(28, 28))
        
        # We use 3 distinct brains for the Ensemble
        cortex_A = QuantumCortex(3136, 10, NEURONS_PER_CLASS, config=physics_config)
        cortex_B = QuantumCortex(3136, 10, NEURONS_PER_CLASS, config=physics_config)
        cortex_C = QuantumCortex(3136, 10, NEURONS_PER_CLASS, config=physics_config)
        
        # --- PHASE 1: TRAINING ---
        print(f"\n=== PHASE 1: TRAINING ({TRAIN_SAMPLES} samples) ===")
        correct_count = 0
        start_time = time.time()
        
        for i in range(TRAIN_SAMPLES):
            img_2d = train_images[i].reshape(28, 28)
            features = optics.apply(img_2d)
            label = train_labels[i]
            
            # Train (Plasticity ON)
            _, pred_a, _ = cortex_A.process_image(features, label, train=True)
            _, pred_b, _ = cortex_B.process_image(features, label, train=True)
            _, pred_c, _ = cortex_C.process_image(features, label, train=True)
            
            # Voting
            votes = np.zeros(10)
            votes[pred_a] += 1; votes[pred_b] += 1; votes[pred_c] += 1
            ensemble_pred = np.argmax(votes)
            
            if ensemble_pred == label: correct_count += 1
            
            # Annealing
            if (i + 1) % 1000 == 0:
                progress = i / TRAIN_SAMPLES
                cortex_A.decay_learning_rate(progress)
                cortex_B.decay_learning_rate(progress)
                cortex_C.decay_learning_rate(progress)
                
                acc = (correct_count / (i + 1)) * 100
                print(f"Train {i+1} | Acc: {acc:.2f}% | Vote: {ensemble_pred}/{label}")

        train_acc = (correct_count / TRAIN_SAMPLES) * 100
        print(f"Training Complete. Final Train Acc: {train_acc:.2f}%")

        # --- PHASE 2: TESTING ---
        print(f"\n=== PHASE 2: VALIDATION (10,000 samples) ===")
        print("Plasticity OFF. Testing Generalization...")
        
        test_correct = 0
        
        for i in range(len(test_images)):
            img_2d = test_images[i].reshape(28, 28)
            features = optics.apply(img_2d)
            label = test_labels[i]
            
            # Test (Plasticity OFF)
            _, pred_a, _ = cortex_A.process_image(features, label, train=False)
            _, pred_b, _ = cortex_B.process_image(features, label, train=False)
            _, pred_c, _ = cortex_C.process_image(features, label, train=False)
            
            votes = np.zeros(10)
            votes[pred_a] += 1; votes[pred_b] += 1; votes[pred_c] += 1
            ensemble_pred = np.argmax(votes)
            
            if ensemble_pred == label: test_correct += 1
            
            if (i + 1) % 1000 == 0:
                print(f"Test {i+1} | Current Test Acc: {(test_correct / (i+1))*100:.2f}%")

        test_acc = (test_correct / len(test_images)) * 100
        total_time = time.time() - start_time
        
        print(f"\n=== FINAL RESULTS ===")
        print(f"Training Accuracy: {train_acc:.2f}%")
        print(f"Test Accuracy:     {test_acc:.2f}%")
        
        log_experiment(train_acc, test_acc, total_time, physics_config, "Validation Run")

    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    run_validation()