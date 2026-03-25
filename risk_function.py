import numpy as np

fusion = FusionEngine()

def detect_threat(network_input, process_input, net_window=[]):

    # --- 1. Network model ---
    reconstructed = autoencoder.predict(network_input)
    net_error = np.mean(np.square(network_input - reconstructed), axis=1)[0]

    # keep last few scores (window)
    net_window.append(net_error)
    if len(net_window) > 10:
        net_window.pop(0)

    # --- 2. Process model ---
    process_scaled = scaler.transform(process_input)
    process_scaled = process_scaled.reshape((1, process_scaled.shape[0], process_scaled.shape[1]))

    prediction = lstm_model.predict(process_scaled)
    proc_error = np.mean(np.square(process_scaled - prediction))

    # --- 3. Fusion ---
    result = fusion.fuse(net_window, proc_error)

    return result