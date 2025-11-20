import time
import csv
import os
import numpy as np
import random
from mnist_loader import LocalMNISTLoader
from quantum_phase_snn import QuantumPhaseSNN
from fourier_optics import GaborFilterBank

def log_monte_carlo(run_id, config, accuracy, duration):
    filename = "monte_carlo_optics.csv"
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Log the optical parameters
            writer.writerow(["Run_ID", "Accuracy", "Duration", "K_Size", "Sigma", "Lambda", "Gamma"])
        
        writer.writerow([
            run_id,
            f"{accuracy:.2f}%",
            f"{duration:.1f}s",
            config['k_size'],
            f"{config['sigma']:.2f}",
            f"{config['lambd']:.2f}",
            f"{config['gamma']:.2f}"
        ])
    print(f"[MC] Result logged: {accuracy:.2f}%")

def run_optics_tuning():
    data_path = r"C:\Users\Maxwell Wilson\OneDrive\Documents\Quantum_snn_benchmark\mnist_data"
    loader = LocalMNISTLoader(data_path)
    
    # Load Data Once
    print("--- Initializing Gabor Optics Tuning ---")
    images_all = loader.load_images('train-images.idx3-ubyte')
    labels_all = loader.load_labels('train-labels.idx1-ubyte')
    
    # CONFIG
    NUM_TRIALS = 30       
    SAMPLES_PER_TRIAL = 2000 # Short sprints
    NEURONS_PER_CLASS = 5   # Keep it light for tuning
    
    for run_id in range(1, NUM_TRIALS + 1):
        # 1. Randomize Gabor Physics
        # K_size: Window size (3, 5, 7, 9). Larger = captures larger features.
        # Sigma: Spread. 
        # Lambda: Wavelength. 
        # Gamma: Aspect ratio.
        
        k_choice = random.choice([5, 7, 9])
        
        optics_config = {
            'k_size': k_choice, 
            'sigma':  random.uniform(0.5, k_choice/2),
            'lambd':  random.uniform(1.5, k_choice),
            'gamma':  random.uniform(0.3, 1.0)
        }
        
        print(f"\n--- Trial {run_id}/{NUM_TRIALS} ---")
        print(f"Optics: K={optics_config['k_size']} | Sig={optics_config['sigma']:.2f} | Lam={optics_config['lambd']:.2f} | Gam={optics_config['gamma']:.2f}")
        
        # 2. Initialize Optics with new settings
        try:
            optics = GaborFilterBank(config=optics_config)
        except:
            print("Invalid Gabor params, skipping...")
            continue

        # 3. Initialize Brain (Network params are locked to 77.6% winners)
        snn = QuantumPhaseSNN(
            num_inputs=784 * 4, # Gabor always outputs 4 channels 
            num_classes=10, 
            neurons_per_class=NEURONS_PER_CLASS
        )
        
        # 4. Shuffle Data
        indices = np.arange(len(images_all))
        np.random.shuffle(indices)
        images = images_all[indices[:SAMPLES_PER_TRIAL]]
        labels = labels_all[indices[:SAMPLES_PER_TRIAL]]
        
        correct_count = 0
        start_time = time.time()
        
        # 5. Run Sprint
        for i in range(SAMPLES_PER_TRIAL):
            # Diffraction
            img_2d = images[i].reshape(28, 28)
            features = optics.apply(img_2d)
            
            # Quantum Processing
            is_correct, pred = snn.process_image(features, labels[i], train=True)
            if is_correct: correct_count += 1
            
            if (i+1) % 1000 == 0:
                print(f"   Step {i+1}: {(correct_count/(i+1))*100:.2f}%")

        total_time = time.time() - start_time
        final_acc = (correct_count / SAMPLES_PER_TRIAL) * 100
        
        # 6. Log
        log_monte_carlo(run_id, optics_config, final_acc, total_time)

if __name__ == "__main__":
    run_optics_tuning()