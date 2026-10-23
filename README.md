# README.md: Automatic Modulation Classification (AMC) under Simulated Channel Impairments

Deep-learning classification of 9 digital modulation types from raw I/Q samples[cite: 21]. This project evaluates and compares a baseline 1D Convolutional Neural Network (CNN) against a proposed hybrid architecture (CNN + BiLSTM + Transformer) using a custom-generated dataset[cite: 21].

| Model | Best Validation Accuracy | Test Accuracy |
| :--- | :---: | :---: |
| **Given CNN (Baseline)** | 77.24%[cite: 21] | 78.44%[cite: 21] |
| **Proposed CNN + BiLSTM + Transformer** | 87.99%[cite: 21] | **87.43%**[cite: 21] |

---

## 1. Project Overview
Automatic Modulation Classification (AMC) identifies the modulation scheme of received radio signals without requiring prior knowledge of the transmitter[cite: 21]. It is a critical component in spectrum sensing and cognitive radio operations[cite: 21]. 

This project encompasses:
* Generating a synthetic dataset of 9 modulation classes subjected to root-raised-cosine (RRC) pulse shaping, multipath fading, Doppler spread, carrier frequency offset (CFO), and AWGN[cite: 21].
* Training two distinct deep learning architectures on identical data splits[cite: 21].
* Evaluating performance metrics, including overall test accuracy, per-class behavior via confusion matrices, and accuracy versus Signal-to-Noise Ratio (SNR)[cite: 21].

---

## 2. Dataset Specifications
The dataset is generated via `Dataset.py` and stored as `modulation_dataset2.h5`[cite: 21]. 

* **Total Examples:** 49,500 frames[cite: 21].
* **Input Shape:** `(2, 1024)` representing 1,024 complex samples separated into I and Q channels[cite: 21].
* **SNR Range:** 11 distinct levels from -1 dB to 19 dB (in 2 dB increments)[cite: 21].
* **Data Split:** Randomly partitioned into 70% Training (34,650), 15% Validation (7,425), and 15% Testing (7,425)[cite: 21].

### 2.1 Modulation Classes
* **Amplitude:** OOK, ASK[cite: 21].
* **Phase:** BPSK, QPSK, 8PSK, 16PSK[cite: 21].
* **Frequency:** BFSK, 4FSK, 8FSK[cite: 21].

### 2.2 Channel Impairments
Signals are progressively distorted to simulate real-world conditions[cite: 21]:
1. **Multipath Fading:** Time-varying Rayleigh fading with an exponential power-delay profile (2-4 taps)[cite: 21].
2. **Doppler Spread:** Independent Doppler shifts applied per tap[cite: 21].
3. **Carrier Frequency Offset (CFO):** Uniform rotation up to ±0.01 cycles/sample, which induces up to 10 full carrier rotations per frame[cite: 21]. This severe phase smearing is the primary difficulty factor for phase-only classes[cite: 21].
4. **AWGN:** Applied based on target SNR parameters[cite: 21].

---

## 3. Model Architectures
Both models accept normalized unit average power I and Q channels as inputs[cite: 21]. They are trained using Cross-entropy loss and the Adam optimizer (learning rate = $10^{-3}$) over 15 epochs, selecting the weights that yield the best validation accuracy[cite: 21].

### 3.1 Given Model: 1D CNN (`Given_Model.py`)
* Extracts features using two 1D convolutional layers with Batch Normalization, ReLU activations, and Max Pooling[cite: 21].
* Flattens the feature map into dense linear layers (128 and 64 units) with Dropout (0.3)[cite: 21].
* **Limitation:** Flattening ties weights to fixed time positions, restricting the model's ability to relate distant temporal symbols[cite: 21].

### 3.2 Proposed Model: CNN + BiLSTM + Transformer (`Proposed_Model.py`)
* **CNN Front-End:** Uses identical 1D convolutional layers to extract local, matched-filter-like patterns[cite: 21].
* **BiLSTM:** A bidirectional LSTM (1 layer, 128 hidden units) tracks feature evolution forwards and backwards across time[cite: 21].
* **Transformer Encoder:** Applies global multi-head self-attention (1 layer, 8 heads, 512 feed-forward size) across the entire sequence window[cite: 21].
* **Classification Head:** Employs Adaptive Average Pooling to summarize the sequence context before the final dense layers, preserving temporal flexibility[cite: 21].

---

## 4. Results & Analysis
* **Overall Accuracy:** The proposed hybrid model achieved an 87.43% test accuracy, outperforming the CNN baseline by +8.99 percentage points[cite: 21]. 
* **Training Dynamics:** The CNN's training loss plateaued around 0.34[cite: 21]. The proposed model's loss decreased to 0.19-0.20, and its validation accuracy was still rising at epoch 14, indicating training may have stopped early[cite: 21].
* **Class-Specific Shifts:**
    * **Improvements:** The overall gain was driven entirely by lower-order phase modulations. QPSK accuracy surged by +71.1 points (20.8% to 91.9%), and 8PSK improved by +35.0 points (55.9% to 90.9%)[cite: 21].
    * **Degradations:** 16PSK classification suffered dramatically, falling by 23.8 points (from 29.5% to 5.7%)[cite: 21]. The proposed model overwhelmingly misclassified 16PSK examples as 8PSK (740 out of 819 instances)[cite: 21].

---

## 5. Future Work
To address the current limitations and expand the model's capabilities, the following areas have been identified for future development:
* **Dataset Expansion:** Introduce higher-order modulations such as QAM, and evaluate the architecture against standardized RadioML datasets and real over-the-air signal captures[cite: 9].
* **Addressing Phase Confusion:** Target the persistent confusion between QPSK, 8PSK, and 16PSK by incorporating constellation-based inputs, higher-order statistics, or implementing deeper attention stacks[cite: 9].
* **Architecture Ablation:** Run strict ablation studies separating the BiLSTM from the Transformer to isolate their individual contributions[cite: 9]. 
* **Evaluation Rigor:** Test both models using identical, globally seeded data splits and accurately measure inference times[cite: 9, 21].
* **Optimization:** Compress the hybrid model to facilitate deployment on resource-constrained hardware[cite: 9].
