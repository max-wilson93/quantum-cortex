import time
import csv
import os
import numpy as np
from mnist_loader import LocalMNISTLoader
from quantum_cortex import QuantumCortex
from fourier_optics import FourierOptics

def log_experiment(accuracy, duration, n_samples, neurons, notes):
    filename = "quantum_training_log.csv"
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Timestamp", "Accuracy", "Duration", "Samples", "Neurons", "Notes"])
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), f"{accuracy:.2f}%", f"{duration:.1f}s", n_samples, neurons, notes])
    print(f"\n[Log] Saved to {filename}")

def run_benchmark():
    data_path = r"C:\Users\Maxwell Wilson\OneDrive\Documents\Quantum_snn_benchmark\quantum-cortex\mnist_data"
    loader = LocalMNISTLoader(data_path)
    
    NEURONS_PER_CLASS = 4
    N_SAMPLES = 60000 
    
    try:
        print("--- Quantum Cortex (Unitary L2 Normalization) ---")
        images = loader.load_images('train-images.idx3-ubyte')
        labels = loader.load_labels('train-labels.idx1-ubyte')
        
        indices = np.arange(len(images))
        np.random.shuffle(indices)
        images = images[indices]
        labels = labels[indices]
        
        print("-> Initializing Fourier Lens...")
        optics = FourierOptics(shape=(28, 28))
        
        print("-> Initializing Cortex...")
        snn = QuantumCortex(num_inputs=3136, num_classes=10, neurons_per_class=NEURONS_PER_CLASS)
        
        correct_count = 0
        start_time = time.time()
        
        for i in range(N_SAMPLES):
            img_2d = images[i].reshape(28, 28)
            features = optics.apply(img_2d)
            
            # Unpacking 3 values
            is_correct, pred, energy = snn.process_image(features, labels[i], train=True)
            
            if is_correct: correct_count += 1
            
            if (i + 1) % 1000 == 0:
                acc = (correct_count / (i + 1)) * 100
                # Energy should be clamped near 10.0 (System Energy)
                print(f"Sample {i+1} | Acc: {acc:.2f}% | Pred: {pred}/{labels[i]} | Energy: {energy:.2f}")
                

        total_time = time.time() - start_time
        final_acc = (correct_count/N_SAMPLES)*100
        
        print(f"\nCortex Run Complete.")
        print(f"Final Accuracy: {final_acc:.2f}%")
        
        log_experiment(final_acc, total_time, N_SAMPLES, NEURONS_PER_CLASS, "Cortex: Unitary L2 (Fixed Return)")

    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    run_benchmark()