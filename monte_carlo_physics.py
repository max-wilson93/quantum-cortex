import time
import csv
import os
import numpy as np
import random
from mnist_loader import LocalMNISTLoader
from quantum_cortex import QuantumCortex
from fourier_optics import FourierOptics

def log_monte_carlo(run_id, config, accuracy, duration):
    filename = "monte_carlo_physics.csv"
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Header
            writer.writerow(["Run_ID", "Accuracy", "Duration", "LR", "Flex", "Lat_Str", "Thresh", "Kerr", "Sys_E"])
        
        writer.writerow([
            run_id,
            f"{accuracy:.2f}%",
            f"{duration:.1f}s",
            f"{config['learning_rate']:.5f}",
            f"{config['phase_flexibility']:.4f}",
            f"{config['lateral_strength']:.4f}",
            f"{config['input_threshold']:.2f}",
            f"{config['kerr_constant']:.2f}",
            f"{config['system_energy']:.1f}"
        ])
    print(f"[MC] Run {run_id} Logged: {accuracy:.2f}%")

def run_optimization():
    data_path = r"./mnist_data" # Ubuntu/Linux Path
    # data_path = r"C:\Users\Maxwell Wilson\OneDrive\Documents\Quantum_snn_benchmark\mnist_data" # Windows Path
    
    loader = LocalMNISTLoader(data_path)
    
    print("--- Initializing Quantum Monte Carlo (Sequential Stability) ---")
    images_all = loader.load_images('train-images.idx3-ubyte')
    labels_all = loader.load_labels('train-labels.idx1-ubyte')
    
    optics = FourierOptics(shape=(28, 28))
    
    # --- CONFIGURATION ---
    NUM_TRIALS = 1000       
    SAMPLES_PER_TRIAL = 10000 
    NEURONS_PER_CLASS = 5    
    
    for run_id in range(0, NUM_TRIALS + 1):
        # --- RUN 0: CONTROL GROUP ---
        if run_id == 0:
            config = {
                'learning_rate':    0.025,
                'phase_flexibility': 0.125,
                'lateral_strength':  0.2,
                'input_threshold':   0.35,
                'kerr_constant':     0.5,
                'system_energy':     10.0
            }
            print("\n--- CONTROL RUN (Baseline) ---")
        else:
            # --- RANDOMIZED TRIALS ---
            config = {
                'learning_rate':    10 ** random.uniform(-3, -1), 
                'phase_flexibility': random.uniform(0.05, 0.30),
                'lateral_strength':  random.uniform(0.0, 0.50),
                'input_threshold':   random.uniform(0.20, 0.50),
                'kerr_constant':     10 ** random.uniform(-1, 1), 
                'system_energy':     random.uniform(5.0, 50.0)
            }
            print(f"\n--- Trial {run_id}/{NUM_TRIALS} ---")
        
        print(str(config))
        
        # Shuffle
        indices = np.arange(len(images_all))
        np.random.shuffle(indices)
        images = images_all[indices[:SAMPLES_PER_TRIAL]]
        labels = labels_all[indices[:SAMPLES_PER_TRIAL]]
        
        # Initialize Brain
        snn = QuantumCortex(
            num_inputs=3136, 
            num_classes=10, 
            neurons_per_class=NEURONS_PER_CLASS,
            config=config
        )
        
        correct_count = 0
        start_time = time.time()
        
        # Run Sprint
        for i in range(SAMPLES_PER_TRIAL):
            img_2d = images[i].reshape(28, 28)
            features = optics.apply(img_2d)
            
            # Unpack 3 values (Compatible with current Cortex)
            is_correct, pred, _ = snn.process_image(features, labels[i], train=True)
            
            if is_correct: correct_count += 1
            
            if (i+1) % 2500 == 0:
                acc = (correct_count / (i+1)) * 100
                print(f"   Step {i+1}: {acc:.2f}%")

        total_time = time.time() - start_time
        final_acc = (correct_count / SAMPLES_PER_TRIAL) * 100
        
        log_monte_carlo(run_id, config, final_acc, total_time)

if __name__ == "__main__":
    run_optimization()