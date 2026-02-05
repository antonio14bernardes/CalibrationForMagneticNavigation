import numpy as np

def rse(true_fields=None, pred_fields=None, eps=1e-12):
    """
    Metrics:
      - rse:        per-sample ||pred - true||                  (shape: (N,))
      - rse_rel:    per-sample ||pred - true|| / ||true||       (shape: (N,))
      - rmse:       scalar sqrt(mean((pred - true)^2))          (float)
    """

    if true_fields is None and pred_fields is None:
        return ["rse", "rse_rel", "rmse"]

    true = np.asarray(true_fields, dtype=np.float64)
    pred = np.asarray(pred_fields, dtype=np.float64)

    # Error vector per sample
    err = pred - true                      # (N, 3)

    # RSE
    rse = np.linalg.norm(err, axis=1)      # (N,)

    # Relative RSE
    true_mag = np.linalg.norm(true, axis=1)        # (N,)
    rse_rel = rse / (true_mag + eps)               # (N,)

    # RMSE: scalar
    rmse = float(np.sqrt(np.mean(rse**2)))         # single float

    return {
        "rse": rse,
        "rse_rel": rse_rel,
        "rmse": rmse,
    }

def mag_and_angle(true_fields=None, pred_fields=None, eps=1e-12):
    """
    Compute:
      - mag_rel: relative magnitude error per sample
                 | ||pred|| - ||true|| | / (||true|| + eps)
      - angle:   angle between true and pred in degrees per sample
    Returns a dict to be merged into the model metrics.
    """

    if true_fields is None and pred_fields is None:
        return ["mag_rel", "angle"]

    true = np.asarray(true_fields, dtype=np.float64)
    pred = np.asarray(pred_fields, dtype=np.float64)

    # Magnitudes
    true_mag = np.linalg.norm(true, axis=1)   # (N,)
    pred_mag = np.linalg.norm(pred, axis=1)   # (N,)

    # Relative magnitude error
    mag_rel = np.abs(pred_mag - true_mag) / (true_mag + eps)  # (N,)

    # Angle in degrees
    dot = np.sum(true * pred, axis=1)        # (N,)
    denom = true_mag * pred_mag + eps
    cos_theta = dot / denom
    cos_theta = np.clip(cos_theta, -1.0, 1.0)   # numerical safety
    angle = np.degrees(np.arccos(cos_theta))    # (N,)

    return {
        "mag_rel": mag_rel,
        "angle": angle,
    }


def grad_curl_div(true_fields=None, pred_fields=None):
    """
    Compute:
      - curl magnituce:     per-sample ||curl(pred - true)||        (shape: (N,))
      - div:                per-sample div(pred - true)             (shape: (N,))
    Returns a dict to be merged into the model metrics.
    """

    if true_fields is None and pred_fields is None:
        return ["curl_mag", "div"]

    # Check shape
    if pred_fields.shape[1] != 3 or pred_fields.shape[2] != 3:
        raise ValueError("Input grads must have shape (N, 3, 3)")
    
    # Compute divergence
    div = pred_fields[:, 0, 0] + \
          pred_fields[:, 1, 1] + \
          pred_fields[:, 2, 2]        # (N,)
    
    # Compute curl
    curl_x = pred_fields[:, 2, 1] - pred_fields[:, 1, 2]
    curl_y = pred_fields[:, 0, 2] - pred_fields[:, 2, 0]
    curl_z = pred_fields[:, 1, 0] - pred_fields[:, 0, 1]
    curl = np.sqrt(curl_x**2 + curl_y**2 + curl_z**2)          # (N,)

    return {
        "curl": curl,
        "div": div,
    }