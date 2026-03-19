# api/validators.py

REQUIRED_FIELDS = ["time_step"]
MIN_FEATURES    = 10   # minimum number of f* feature fields


def validate_predict_request(data: dict) -> tuple[bool, str | None]:
    """
    Validate a single prediction request body.
    Returns (is_valid, error_message).
    """
    if not data:
        return False, "Request body is empty or not valid JSON."

    for field in REQUIRED_FIELDS:
        if field not in data:
            return False, f"Missing required field: {field}"

    time_step = data.get("time_step")
    if not isinstance(time_step, (int, float)) or not (1 <= time_step <= 49):
        return False, "time_step must be a number between 1 and 49."

    feature_keys = [k for k in data.keys() if k.startswith("f")]
    if len(feature_keys) < MIN_FEATURES:
        return False, (f"At least {MIN_FEATURES} feature fields "
                       f"(f1, f2, …) are required.")

    for key in feature_keys:
        if not isinstance(data[key], (int, float)):
            return False, f"Feature '{key}' must be a numeric value."

    return True, None


def validate_batch_request(data: dict,
                            max_batch: int = 100) -> tuple[bool, str | None]:
    """Validate a batch prediction request."""
    if not data or "transactions" not in data:
        return False, "Request must contain a 'transactions' list."

    txns = data["transactions"]
    if not isinstance(txns, list) or len(txns) == 0:
        return False, "'transactions' must be a non-empty list."

    if len(txns) > max_batch:
        return False, f"Batch size cannot exceed {max_batch} transactions."

    return True, None
