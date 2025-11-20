import time
import csv
import os
import numpy as np
from mnist_loader import LocalMNISTLoader
from quantum_cortex import QuantumCortex
from fourier_optics import FourierOptics

def run_benchmark():
    data_path = r"./mnist_data"
    loader = LocalMNISTLoader(data_path)
    
    NEURONS_PER_CLASS = 20
    N_SAMPLES = 30000
    
    try:
        print("--- Quantum Cortex (Tanh Saturation Verification) ---")
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
    
            is_correct, pred, energy = snn.process_image(features, labels[i], train=True)

            if is_correct: correct_count += 1
            
            if (i + 1) % 1000 == 0:
                acc = (correct_count / (i + 1)) * 100
                print(f"Sample {i+1} | Acc: {acc:.2f}% | Pred: {pred}/{labels[i]}")
                
            if (i + 1) % 10000 == 0:
                snn.visualize_cortex_ascii(0)

        final_acc = (correct_count/N_SAMPLES)*100
        print(f"Final Accuracy: {final_acc:.2f}%")

    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    run_benchmark()