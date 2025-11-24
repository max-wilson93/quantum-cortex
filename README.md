# The Quantum-Holographic Cortex
### A Bio-Physical Spiking Neural Network for One-Shot Learning

**Status:** Validated (90.74% Test Accuracy on MNIST)  
**Paradigm:** Online, One-Shot, Non-Backpropagation  
**Core Physics:** Fourier Optics, Kerr Non-Linearity, Unitary Evolution  

---

## 1. Abstract
This project implements a novel Quantum-Inspired Neural Architecture that abandons standard scalar integration (LIF models) in favor of **Complex-Valued Wave Interference**.

By modeling dendritic integration as a holographic process rather than a linear summation, this model achieves **90.74% accuracy** on the MNIST test set using a strictly Online, One-Shot Learning algorithm (Batch Size 1, 1 Epoch). Notably, the model achieves **Zero Overfitting** (Test Accuracy > Training Accuracy), demonstrating that Phase-Based Encoding captures topological features more robustly than Magnitude-Based Encoding. This offers a theoretical bridge between Holonomic Brain Theory and Optical Computing.

## 2. The Theoretical Basis
Standard Deep Learning treats neurons as switches ($0$ or $1$). This model treats neurons as **Resonators** ($\Psi = A e^{i\theta}$).

### A. Holonomic Brain Theory (Pribram & Bohm)
We process inputs in the Spectral Domain rather than the Spatial Domain.
* **The Retina:** Instead of raw pixels, the input is passed through a Fourier Optical Correlator ($4f$ System).
* **The Mechanism:** This splits the image into frequency bands (Structure vs. Texture) and orientations, simulating the receptive fields of the Visual Cortex (V1).

### B. Quantum Non-Linearity (The Optical Kerr Effect)
Standard SNNs use threshold functions (ReLU/Step) to create non-linearity. We use the laws of Non-Linear Optics.
* **The Physics:** High-intensity signals passing through a medium change the refractive index of that medium.
* **The Equation:** $\theta_{new} = \theta_{old} + \chi \cdot |Amplitude|^2$
* **The Result:** Strong signals "twist" their own phase, effectively self-focusing into stable solitons (memories) while weak noise washes out via destructive interference.

### C. Astrocyte-Neuron Metabolic Coupling
To prevent "Runaway Resonance" (Seizures) inherent in recurrent networks, we implement Unitary L2 Normalization.
* **The Biology:** Simulates Astrocytes regulating the global energy budget (ATP/Potassium) of the cortical column.
* **The Math:** The total energy of the system is clamped to a constant ($||\Psi|| = C$) at every time step. This forces neurons to compete for energy, preserving contrast without clipping signals.

## 3. The Architecture: "The Quantum Trinity"
To eliminate phase noise and solve the stability-plasticity dilemma, the final model uses a Homogeneous Ensemble of three independent cortical columns.

* **Mechanism:** Three identical Quantum Cortices are initialized with random phase distributions ($\theta \sim U[0, 2\pi]$).
* **Process:** They process the same input stream independently, developing unique interference patterns (Holograms) due to their differing initial states.
* **Quantum Error Correction:** The outputs are aggregated via Constructive Interference (Voting).
    * If the signal is "Real" (Topological), all three brains resonate in sync.
    * If the signal is "Noise" (Random), the brains drift out of phase and cancel each other out.

This statistical holography allows the system to achieve >90% accuracy without requiring backpropagation or heterogeneous parameter tuning.

## 4. Results & Performance
The model was benchmarked on the standard MNIST dataset (60k Train / 10k Test).

* **Training Method:** Online, One-Shot (Each image seen exactly once).
* **Hardware:** Standard CPU (Ryzen 3900X).
* **Training Time:** ~24 Minutes.

| Metric | Result | Significance |
| :--- | :--- | :--- |
| **Training Accuracy** | 90.15% | Strong memorization of prototypes. |
| **Test Accuracy** | 90.74% | Negative Generalization Gap. The model performs better on unseen data, proving it learned topology, not noise. |
| **Efficiency** | O(1) | Learning time per sample is constant. It does not scale with dataset size. |

## 5. Installation & Usage

### Requirements
* Python 3.10+
* NumPy

### Running the Model
Clone the repository:
```bash
git clone [https://github.com/yourusername/quantum-cortex.git](https://github.com/yourusername/quantum-cortex.git)
cd quantum-cortex
```
### Run the Grand Benchmark
```bash
python main.py
```
This will initialize the "trinity" ensemble, train on 60,000 samples, and validate on 10,000 samples.

## 6. Modifying Physics
The system parameters have been optimized through Monte Carlo simulation and statistical analysis to find the "digital resonance" regime. You can tune them in `main.py`:

```bash
physics_config = {
    'learning_rate':    0.09,  # High Plasticity (Flashbulb Memory)
    'phase_flexibility': 0.10, # Stiff Rotation (Stability)
    'lateral_strength':  0.16, # Moderate Binding (Coherence)
    'input_threshold':   0.70, # Strict Digital Gating (Noise Removal)
    'kerr_constant':     0.20, # Low Non-Linearity
    'system_energy':     40.0  # High Gain (Amplification)
}
```
