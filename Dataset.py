import numpy as np
import h5py
import multiprocessing as mp
from scipy.interpolate import interp1d

# =====================================================
# Parameters
# =====================================================

NUM_SAMPLES = 128
SAMPLES_PER_SYMBOL = 8
NUM_EXAMPLES = 500

# SNR in standard dB scale, from -1 to 20 in steps of 1
SNR_VALUES = np.arange(-1, 21, 1)

MODULATIONS = [
    "OOK", "ASK", "BPSK", "QPSK", "8PSK", 
    "16PSK", "BFSK", "4FSK", "8FSK"
]

LABEL_MAP = {mod: i for i, mod in enumerate(MODULATIONS)}

# =====================================================
# Pulse Shaping: Root Raised Cosine (RRC)
# =====================================================

RRC_BETA = 0.35
RRC_SPAN = 6

def rrc_filter(beta, span, sps):
    N = span * sps
    t = (np.arange(-N / 2, N / 2 + 1)) / sps
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            h[i] = 1.0 - beta + 4 * beta / np.pi
        elif beta != 0 and np.isclose(abs(ti), 1 / (4 * beta)):
            h[i] = (beta / np.sqrt(2)) * (((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))) + ((1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))))
        else:
            h[i] = (np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))) / (np.pi * ti * (1 - (4 * beta * ti) ** 2))
    return h / np.sqrt(np.sum(h ** 2))

_RRC_TAPS = rrc_filter(RRC_BETA, RRC_SPAN, SAMPLES_PER_SYMBOL)
_FILTER_DELAY = (len(_RRC_TAPS) - 1) // 2
_GEN_SYMBOLS = (NUM_SAMPLES + _FILTER_DELAY * 2) // SAMPLES_PER_SYMBOL + 2

def pulse_shape(symbols):
    upsampled = np.zeros(len(symbols) * SAMPLES_PER_SYMBOL, dtype=complex)
    upsampled[::SAMPLES_PER_SYMBOL] = symbols
    shaped = np.convolve(upsampled, _RRC_TAPS)
    start = _FILTER_DELAY
    return shaped[start:start + NUM_SAMPLES]

# =====================================================
# Modulation Generators
# =====================================================

def ook():
    bits = np.random.randint(0, 2, _GEN_SYMBOLS)
    return pulse_shape(bits.astype(complex))

def ask():
    bits = np.random.randint(0, 2, _GEN_SYMBOLS)
    amplitudes = (2 * bits + 1).astype(complex)
    return pulse_shape(amplitudes)

def bpsk():
    bits = np.random.randint(0, 2, _GEN_SYMBOLS)
    return pulse_shape((2 * bits - 1).astype(complex))

def qpsk():
    symbols_idx = np.random.randint(0, 4, _GEN_SYMBOLS)
    phase = np.pi / 4 + symbols_idx * np.pi / 2
    return pulse_shape(np.exp(1j * phase))

def psk8():
    symbols_idx = np.random.randint(0, 8, _GEN_SYMBOLS)
    phase = symbols_idx * 2 * np.pi / 8
    return pulse_shape(np.exp(1j * phase))

def psk16():
    symbols_idx = np.random.randint(0, 16, _GEN_SYMBOLS)
    phase = symbols_idx * 2 * np.pi / 16
    return pulse_shape(np.exp(1j * phase))

def _continuous_phase_fsk(symbols, frequencies, sps=SAMPLES_PER_SYMBOL):
    freq_per_symbol = frequencies[symbols]              
    freq_per_sample = np.repeat(freq_per_symbol, sps)    
    phase = 2 * np.pi * np.cumsum(freq_per_sample)
    return np.exp(1j * phase)[:NUM_SAMPLES]

def bfsk():
    symbols = np.random.randint(0, 2, _GEN_SYMBOLS)
    frequencies = np.array([1, 3]) / SAMPLES_PER_SYMBOL
    return _continuous_phase_fsk(symbols, frequencies)

def fsk4():
    symbols = np.random.randint(0, 4, _GEN_SYMBOLS)
    frequencies = np.array([1, 2, 3, 4]) / SAMPLES_PER_SYMBOL
    return _continuous_phase_fsk(symbols, frequencies)

def fsk8():
    symbols = np.random.randint(0, 8, _GEN_SYMBOLS)
    frequencies = np.arange(1, 9) / SAMPLES_PER_SYMBOL
    return _continuous_phase_fsk(symbols, frequencies)

GENERATORS = {
    "OOK": ook, "ASK": ask, "BPSK": bpsk, "QPSK": qpsk,
    "8PSK": psk8, "16PSK": psk16, "BFSK": bfsk, "4FSK": fsk4, "8FSK": fsk8
}

# =====================================================
# Channel Impairments
# =====================================================

def apply_channel(signal):
    N = len(signal)
    
    # 1. Baseband Power Normalization
    signal = signal / np.sqrt(np.mean(np.abs(signal)**2))
    
    # 2. Fractional Symbol Timing Offset (STO)
    t = np.arange(N)
    delay = np.random.uniform(0.0, 0.99)
    interp_func = interp1d(t, signal, kind='cubic', fill_value="extrapolate")
    signal = interp_func(t - delay)
    
    # 3. Rayleigh Multipath Fading
    num_taps = np.random.randint(2, 5)
    tap_powers = np.exp(-np.arange(num_taps))
    tap_powers /= np.sum(tap_powers)
    faded_signal = np.zeros(N + num_taps - 1, dtype=complex)
    
    for i in range(num_taps):
        h = (np.random.randn() + 1j * np.random.randn()) * np.sqrt(tap_powers[i] / 2)
        faded_signal[i:i+N] += signal * h
        
    signal = faded_signal[:N]
    
    # 4. Carrier Frequency Offset (CFO) & Random Walk Phase Noise
    cfo = np.random.uniform(-0.01, 0.01)
    phase_noise_std = np.random.uniform(1e-4, 1e-3)
    random_walk_phase = np.cumsum(np.random.normal(0, phase_noise_std, N))
    frequency_drift = np.exp(1j * (2 * np.pi * cfo * t + random_walk_phase + np.random.uniform(0, 2*np.pi)))
    
    return signal * frequency_drift

def add_awgn(signal, snr_db):
    # Restored DB to Linear conversion
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    return signal + noise

# =====================================================
# Multiprocessing Worker & Main
# =====================================================

def generate_samples(args):
    modulation, snr_db = args
    gen_func = GENERATORS[modulation]
    label = LABEL_MAP[modulation]
    
    local_X, local_Y, local_SNR = [], [], []
    for _ in range(NUM_EXAMPLES):
        sig = gen_func()
        sig = apply_channel(sig)
        sig = add_awgn(sig, snr_db)
        
        iq = np.vstack((sig.real, sig.imag)).astype(np.float32)
        local_X.append(iq)
        local_Y.append(label)
        local_SNR.append(snr_db)
        
    return local_X, local_Y, local_SNR

if __name__ == "__main__":
    tasks = [(mod, snr) for mod in MODULATIONS for snr in SNR_VALUES]
    
    X, Y, SNR = [], [], []
    
    print("Spawning worker pool for dataset generation...")
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(generate_samples, tasks)
        
    for res_X, res_Y, res_SNR in results:
        X.extend(res_X)
        Y.extend(res_Y)
        SNR.extend(res_SNR)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.int64)
    SNR = np.array(SNR, dtype=np.int32)
    
    # Global Shuffling to lock SNR arrays to the data
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X, Y, SNR = X[indices], Y[indices], SNR[indices]

    print(f"Dataset generated. Shape: {X.shape}")
    
    with h5py.File("modulation_dataset2.h5", "w") as f:
        f.create_dataset("X", data=X)
        f.create_dataset("Y", data=Y)
        f.create_dataset("SNR", data=SNR)
        
    print("Saved successfully to modulation_dataset2.h5")
