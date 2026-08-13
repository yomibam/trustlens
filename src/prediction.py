import pandas as pd


METADATA_COLUMNS = [
    "TransactionID",
]


def get_risk_level(probability: float) -> str:
    if probability < 0.30:
        return "Low Risk"

    if probability < 0.60:
        return "Medium Risk"

    if probability < 0.85:
        return "High Risk"

    return "Critical Risk"


def get_recommendation(probability: float) -> str:
    if probability < 0.30:
        return "Approve transaction"

    if probability < 0.60:
        return "Require additional verification"

    if probability < 0.85:
        return "Hold for manual review"

    return "Block or escalate immediately"


def predict_transactions(model, data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate TrustLens fraud predictions.

    TransactionID is retained as metadata for display and investigation,
    but excluded from the predictive feature matrix.
    """

    if data.empty:
        raise ValueError("The uploaded dataset is empty.")

    # Preserve metadata for the investigation interface
    metadata = pd.DataFrame(index=data.index)

    for column in METADATA_COLUMNS:
        if column in data.columns:
            metadata[column] = data[column]

    # Remove metadata before passing data to the model
    model_input = data.drop(
        columns=METADATA_COLUMNS,
        errors="ignore"
    ).copy()

    expected_features = list(model.feature_name_)

    missing_features = [
        feature
        for feature in expected_features
        if feature not in model_input.columns
    ]

    unexpected_features = [
        feature
        for feature in model_input.columns
        if feature not in expected_features
    ]

    if missing_features:
        raise ValueError(
            "The uploaded file is missing model features. "
            f"Missing feature count: {len(missing_features)}. "
            f"First missing features: {missing_features[:10]}"
        )

    if unexpected_features:
        raise ValueError(
            "The uploaded file contains unexpected model features. "
            f"Unexpected feature count: {len(unexpected_features)}. "
            f"First unexpected features: {unexpected_features[:10]}"
        )

    # Reorder columns exactly as expected by the trained model
    model_input = model_input[expected_features]

    if "TransactionID" in model_input.columns:
        raise ValueError(
        "TransactionID must not be passed to the prediction model."
    )

    if model_input.shape[1] != model.n_features_in_:
        raise ValueError(
        f"Feature count mismatch. "
        f"Model expects {model.n_features_in_} features, "
        f"but received {model_input.shape[1]}."
    )

    if list(model_input.columns) != expected_features:
        raise ValueError(
        "The uploaded model features do not match the expected "
        "feature names and order."
    )

    probabilities = model.predict_proba(model_input)[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    results = metadata.copy()

    # Retain all predictive features for later investigation
    for column in model_input.columns:
        results[column] = model_input[column]

    results["fraud_probability"] = probabilities
    results["predicted_isFraud"] = predictions

    results["risk_level"] = [
        get_risk_level(probability)
        for probability in probabilities
    ]

    results["recommendation"] = [
        get_recommendation(probability)
        for probability in probabilities
    ]

    return results