import time
import csv
import os
import numpy as np
from mnist_loader import LocalMNISTLoader
from quantum_cortex import QuantumCortex
from fourier_optics import FourierOptics

def log_experiment(accuracy, duration, n_samples, neurons, config, notes):
    filename = "quantum_training_log.csv"
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Timestamp", "Accuracy", "Duration", "Samples", "Neurons", "Config", "Notes"])
        
        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            f"{accuracy:.2f}%",
            f"{duration:.1f}s",
            n_samples,
            neurons,
            str(config),
            notes
        ])
    print(f"\n[Log] Saved to {filename}")

def run_benchmark():
    data_path = r"./mnist_data"
    loader = LocalMNISTLoader(data_path)
    
    # --- GRAND TRINITY CONFIG ---
    NEURONS_PER_CLASS = 5
    N_SAMPLES = 60000 
    
    # Physics from MC Run #752 (The Stable Winner)
    physics_config = {
        'learning_rate':    0.06,   # High plasticity
        'phase_flexibility': 0.10,  # Stiff rotation (Stability)
        'lateral_strength':  0.16,  # Moderate coupling
        'input_threshold':   0.55,  # Digital Gate
        'kerr_constant':     0.20,  # LOW KERR (Prevents the wobble!)
        'system_energy':     30.0   # High Energy
    }
    
    try:
        print(f"--- Quantum Trinity (3-Cortex Ensemble) ---")
        print(f"Physics: {physics_config}")
        
        images = loader.load_images('train-images.idx3-ubyte')
        labels = loader.load_labels('train-labels.idx1-ubyte')
        
        indices = np.arange(len(images))
        np.random.shuffle(indices)
        images = images[indices]
        labels = labels[indices]
        
        print("-> Initializing Fourier Lens...")
        optics = FourierOptics(shape=(28, 28))
        
        # INITIALIZE 3 BRAINS
        print(f"-> Initializing 3 Quantum Cortices...")
        cortex_A = QuantumCortex(3136, 10, NEURONS_PER_CLASS) # Config is hardcoded in class, ensure it matches or update class to accept it
        cortex_B = QuantumCortex(3136, 10, NEURONS_PER_CLASS)
        cortex_C = QuantumCortex(3136, 10, NEURONS_PER_CLASS)
        
        # Manually inject the Optimized Physics (since we aren't passing config in init in your current class version)
        # If your QuantumCortex accepts config, pass it. If not, we set it here:
        for brain in [cortex_A, cortex_B, cortex_C]:
            brain.learning_rate = physics_config['learning_rate']
            brain.phase_flexibility = physics_config['phase_flexibility']
            brain.lateral_strength = physics_config['lateral_strength']
            brain.input_threshold = physics_config['input_threshold']
            brain.kerr_constant = physics_config['kerr_constant']
            brain.system_energy = physics_config['system_energy']
        
        correct_count = 0
        start_time = time.time()
        
        print(f"-> Starting Ensemble Training ({N_SAMPLES} samples)...")
        
        for i in range(N_SAMPLES):
            img_2d = images[i].reshape(28, 28)
            features = optics.apply(img_2d)
            label = labels[i]
            
            # 1. Parallel Processing (Train all 3)
            # We ignore the individual 'is_correct' for the main score, we want the VOTED score.
            _, pred_a, _ = cortex_A.process_image(features, label, train=True)
            _, pred_b, _ = cortex_B.process_image(features, label, train=True)
            _, pred_c, _ = cortex_C.process_image(features, label, train=True)
            
            # 2. Quantum Error Correction (Voting)
            votes = np.zeros(10)
            votes[pred_a] += 1
            votes[pred_b] += 1
            votes[pred_c] += 1
            
            ensemble_pred = np.argmax(votes)
            
            if ensemble_pred == label:
                correct_count += 1
            
            # 3. Annealing (Apply to all)
            if (i + 1) % 1000 == 0:
                # Linear decay
                progress = i / N_SAMPLES
                new_lr = 0.06 * (1.0 - progress * 0.95) # Decay to 10%
                
                cortex_A.learning_rate = new_lr
                cortex_B.learning_rate = new_lr
                cortex_C.learning_rate = new_lr
                
                acc = (correct_count / (i + 1)) * 100
                print(f"Sample {i+1} | Ensemble Acc: {acc:.2f}% | Vote: {pred_a},{pred_b},{pred_c} / {label} | LR: {new_lr:.4f}")
            


        total_time = time.time() - start_time
        final_acc = (correct_count/N_SAMPLES)*100
        
        print(f"\nTrinity Run Complete.")
        print(f"Final Accuracy: {final_acc:.2f}%")
        
        log_experiment(final_acc, total_time, N_SAMPLES, NEURONS_PER_CLASS, physics_config, "Grand Run: Quantum Trinity (Ensemble)")

    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    run_benchmark()