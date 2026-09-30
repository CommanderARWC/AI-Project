import numpy as np
import h5py

# =====================================================
# Parameters
# =====================================================

NUM_SAMPLES = 1024
SAMPLES_PER_SYMBOL = 8
NUM_SYMBOLS = NUM_SAMPLES // SAMPLES_PER_SYMBOL

NUM_EXAMPLES = 500  # Per modulation per SNR

SNR_VALUES = np.arange(-1, 21, 2)

MODULATIONS = [
    "OOK",
    "ASK",
    "BPSK",
    "QPSK",
    "8PSK",
    "16PSK",
    "BFSK",
    "4FSK",
    "8FSK"
]

LABEL_MAP = {mod: i for i, mod in enumerate(MODULATIONS)}

# =====================================================
# Pulse Shaping: Root Raised Cosine (RRC)
# =====================================================

RRC_BETA = 0.35   # roll-off factor
RRC_SPAN = 6      # filter span in symbols

def rrc_filter(beta, span, sps):
    """Generate Root Raised Cosine filter taps."""
    N = span * sps
    t = (np.arange(-N / 2, N / 2 + 1)) / sps  
    h = np.zeros_like(t)

    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            h[i] = 1.0 - beta + 4 * beta / np.pi
        elif beta != 0 and np.isclose(abs(ti), 1 / (4 * beta)):
            h[i] = (beta / np.sqrt(2)) * (
                ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)))
                + ((1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
            )
        else:
            numerator = np.sin(np.pi * ti * (1 - beta)) + \
                4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            denominator = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = numerator / denominator

    h = h / np.sqrt(np.sum(h ** 2))  
    return h

_RRC_TAPS = rrc_filter(RRC_BETA, RRC_SPAN, SAMPLES_PER_SYMBOL)
_FILTER_DELAY = (len(_RRC_TAPS) - 1) // 2

def pulse_shape(symbols, sps=SAMPLES_PER_SYMBOL):
    """Upsample and apply RRC pulse shaping."""
    upsampled = np.zeros(len(symbols) * sps, dtype=complex)
    upsampled[::sps] = symbols
    shaped = np.convolve(upsampled, _RRC_TAPS)
    
    start = _FILTER_DELAY
    end = start + NUM_SAMPLES
    return shaped[start:end]

# =====================================================
# Linear/PSK Modulation Functions 
# =====================================================

def ook():
    bits = np.random.randint(0, 2, NUM_SYMBOLS)
    return pulse_shape(bits.astype(complex))

def ask():
    bits = np.random.randint(0, 2, NUM_SYMBOLS)
    amplitudes = (2 * bits + 1).astype(complex)
    return pulse_shape(amplitudes)

def bpsk():
    bits = np.random.randint(0, 2, NUM_SYMBOLS)
    return pulse_shape((2 * bits - 1).astype(complex))

def qpsk():
    symbols_idx = np.random.randint(0, 4, NUM_SYMBOLS)
    phase = np.pi / 4 + symbols_idx * np.pi / 2
    return pulse_shape(np.exp(1j * phase))

def psk8():
    symbols_idx = np.random.randint(0, 8, NUM_SYMBOLS)
    phase = symbols_idx * 2 * np.pi / 8
    return pulse_shape(np.exp(1j * phase))

def psk16():
    symbols_idx = np.random.randint(0, 16, NUM_SYMBOLS)
    phase = symbols_idx * 2 * np.pi / 16
    return pulse_shape(np.exp(1j * phase))

# =====================================================
# FSK Modulation Functions 
# =====================================================

def _continuous_phase_fsk(symbols, frequencies, sps=SAMPLES_PER_SYMBOL):
    freq_per_symbol = frequencies[symbols]              
    freq_per_sample = np.repeat(freq_per_symbol, sps)    
    phase = 2 * np.pi * np.cumsum(freq_per_sample)
    return np.exp(1j * phase)

def bfsk():
    symbols = np.random.randint(0, 2, NUM_SYMBOLS)
    frequencies = np.array([1, 3]) / SAMPLES_PER_SYMBOL
    return _continuous_phase_fsk(symbols, frequencies)

def fsk4():
    symbols = np.random.randint(0, 4, NUM_SYMBOLS)
    frequencies = np.array([1, 2, 3, 4]) / SAMPLES_PER_SYMBOL
    return _continuous_phase_fsk(symbols, frequencies)

def fsk8():
    symbols = np.random.randint(0, 8, NUM_SYMBOLS)
    frequencies = np.arange(1, 9) / SAMPLES_PER_SYMBOL
    return _continuous_phase_fsk(symbols, frequencies)

GENERATORS = {
    "OOK": ook, "ASK": ask, "BPSK": bpsk, "QPSK": qpsk,
    "8PSK": psk8, "16PSK": psk16, "BFSK": bfsk, "4FSK": fsk4, "8FSK": fsk8
}

# =====================================================
# Channel Impairments
# =====================================================

def apply_channel_impairments(signal):
    """Applies time-varying Rayleigh multipath fading, Doppler spread, and CFO."""
    N = len(signal)
    t = np.arange(N)
    
    # 1. Time-Varying Rayleigh Fading with Doppler Spread
    num_taps = np.random.randint(2, 5)
    tap_powers = np.exp(-np.arange(num_taps))
    tap_powers /= np.sum(tap_powers)  # Normalize power profile
    
    # Max normalized Doppler shift (fd * Ts) simulating relative motion
    max_doppler = np.random.uniform(1e-5, 1e-3) 
    
    faded_signal = np.zeros(N + num_taps - 1, dtype=complex)
    
    for i in range(num_taps):
        # Rayleigh distributed amplitude and uniform phase for this specific tap
        h_i = (np.random.randn() + 1j * np.random.randn()) * np.sqrt(tap_powers[i] / 2)
        
        # Independent Doppler shift for this path creates the Doppler Spread
        doppler_i = np.random.uniform(-max_doppler, max_doppler)
        
        # Time-varying tap (rotates continuously over the 1024 samples)
        tap_time_varying = h_i * np.exp(1j * 2 * np.pi * doppler_i * t)
        
        # Apply this specific delayed, fading, and phase-shifting path to the signal
        faded_signal[i:i+N] += signal * tap_time_varying
        
    # Truncate back to original 1024-sample length
    delay = (num_taps - 1) // 2
    signal = faded_signal[delay:delay+N]
    
    # 2. Carrier Frequency Offset (CFO) and Global Phase Offset
    # Simulates mismatched local oscillators
    phase_offset = np.random.uniform(0, 2 * np.pi)
    cfo = np.random.uniform(-0.01, 0.01) 
    frequency_drift = np.exp(1j * (2 * np.pi * cfo * t + phase_offset))
    
    return signal * frequency_drift

def add_awgn(signal, snr_db):
    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    return signal + noise

# =====================================================
# Dataset Generation
# =====================================================

if __name__ == "__main__":
    X = []
    Y = []
    SNR = []

    print("Generating Dataset with Rayleigh Fading, Doppler, & AWGN...\n")

    for modulation in MODULATIONS:
        print(f"Generating {modulation}...")
        generator = GENERATORS[modulation]
        label = LABEL_MAP[modulation]

        for snr in SNR_VALUES:
            for _ in range(NUM_EXAMPLES):
                
                # 1. Baseband Generation
                signal = generator()
                
                # 2. Simulated Channel Effects (Rayleigh + Doppler + CFO)
                signal = apply_channel_impairments(signal)
                
                # 3. Receiver Noise
                signal = add_awgn(signal, snr)

                # Format as (2, 1024) for the CNN
                iq = np.vstack((signal.real, signal.imag))

                X.append(iq.astype(np.float32))
                Y.append(label)
                SNR.append(snr)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.int64)
    SNR = np.array(SNR, dtype=np.int32)

    print("\nDataset Created Successfully")
    print("X Shape :", X.shape)
    print("Y Shape :", Y.shape)
    print("SNR Shape :", SNR.shape)

    with h5py.File("modulation_dataset2.h5", "w") as f:
        f.create_dataset("X", data=X)
        f.create_dataset("Y", data=Y)
        f.create_dataset("SNR", data=SNR)

    print("\nDataset saved as modulation_dataset2.h5")
    print("Done.")
