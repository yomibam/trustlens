"""
Publicly supported feature descriptions for the IEEE-CIS Fraud Detection
dataset and TrustLens-generated assessment fields.

Dataset feature descriptions are intentionally limited to the information
publicly disclosed by the dataset provider. No specific business meaning is
assigned to masked C, D, M, V or id-series variables.

Sources:
1. IEEE-CIS Fraud Detection competition data documentation, Kaggle, 2019.
2. C. McDonald and C. Deotte, "Leveraging Machine Learning to Detect Fraud:
   Tips to Developing a Winning Kaggle Solution," NVIDIA Technical Blog, 2021.
"""

from typing import Dict, Tuple


FeatureDescription = Tuple[str, str]


# Explicit descriptions for documented dataset fields and TrustLens outputs.
FEATURE_DESCRIPTIONS: Dict[str, FeatureDescription] = {
    # Dataset identifiers and target
    "TransactionID": (
        "Transaction ID",
        "Identifier assigned to the transaction."
    ),
    "isFraud": (
        "Ground-Truth Label",
        "Observed dataset label indicating whether the transaction was "
        "recorded as fraudulent."
    ),

    # Main transaction fields
    "TransactionDT": (
        "Transaction Time Difference",
        "Time difference from a given reference datetime. "
        "This is not an actual transaction timestamp."
    ),
    "TransactionAmt": (
        "Transaction Amount",
        "Transaction payment amount in US dollars."
    ),
    "ProductCD": (
        "Product Code",
        "Product code associated with the transaction."
    ),

    # Email fields
    "P_emaildomain": (
        "Purchaser Email Domain",
        "Email domain associated with the purchaser."
    ),
    "R_emaildomain": (
        "Recipient Email Domain",
        "Email domain associated with the recipient."
    ),

    # Device fields
    "DeviceType": (
        "Device Type",
        "Type of device associated with the identity record."
    ),
    "DeviceInfo": (
        "Device Information",
        "Device information associated with the identity record."
    ),

    # TrustLens-generated output fields
    "fraud_probability": (
        "Fraud Probability",
        "Probability assigned by the LightGBM fraud-classification model."
    ),
    "predicted_isFraud": (
        "Predicted Class",
        "Binary class produced by applying the configured classification "
        "threshold to the fraud probability."
    ),
    "risk_level": (
        "Risk Category",
        "Decision-support category assigned by TrustLens using the "
        "configured fraud-probability thresholds."
    ),
    "recommendation": (
        "Recommended Action",
        "Operational guidance assigned by TrustLens based on the configured "
        "risk category."
    ),
}


def _is_numbered_feature(
    feature: str,
    prefix: str,
    minimum: int,
    maximum: int
) -> bool:
    """
    Check whether a feature follows a numbered naming pattern.

    Examples:
        C1 to C14
        V1 to V339
        id_01 to id_38
    """
    if not feature.startswith(prefix):
        return False

    suffix = feature[len(prefix):]

    if not suffix.isdigit():
        return False

    number = int(suffix)
    return minimum <= number <= maximum


def get_feature_description(feature: str) -> FeatureDescription:
    """
    Return a display name and publicly supported description for a feature.

    Specific meanings are not assigned to masked variables. Where the
    provider disclosed only a feature-family description, this function
    returns that broader description.
    """

    # Return an explicitly defined description first.
    if feature in FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS[feature]

    # Payment-card-related fields
    if _is_numbered_feature(feature, "card", 1, 6):
        return (
            feature,
            "Payment-card-related information. The precise field-level "
            "meaning is masked."
        )

    # Address-related fields
    if feature in {"addr1", "addr2"}:
        return (
            feature,
            "Address-related information. The precise field-level meaning "
            "is masked."
        )

    # Distance-related fields
    if feature in {"dist1", "dist2"}:
        return (
            feature,
            "Distance-related information. The precise field-level meaning "
            "is masked."
        )

    # Counting-related fields
    if _is_numbered_feature(feature, "C", 1, 14):
        return (
            feature,
            "An anonymised C-series counting feature. Its precise meaning "
            "is masked."
        )

    # Time-difference fields
    if _is_numbered_feature(feature, "D", 1, 15):
        return (
            feature,
            "An anonymised D-series time-difference feature. Its precise "
            "meaning is masked."
        )

    # Match-related fields
    if _is_numbered_feature(feature, "M", 1, 9):
        return (
            feature,
            "An anonymised M-series match feature. Its precise meaning "
            "is masked."
        )

    # Vesta-engineered fields
    if _is_numbered_feature(feature, "V", 1, 339):
        return (
            feature,
            "A masked Vesta-engineered feature involving ranking, counting "
            "or other entity relationships. Its precise meaning is not "
            "publicly disclosed."
        )

    # Identity fields
    if _is_numbered_feature(feature, "id_", 1, 38):
        return (
            feature,
            "A masked identity-related feature. Its precise meaning is not "
            "publicly disclosed."
        )

    # Handle one-hot or otherwise encoded versions of documented features.
    # Examples: ProductCD_C, card4_visa, DeviceType_mobile
    encoded_prefixes = {
        "ProductCD_": (
            "Encoded Product Code",
            "An encoded category derived from the transaction product code."
        ),
        "card1_": (
            "Encoded Card Feature",
            "An encoded category derived from masked payment-card-related "
            "information."
        ),
        "card2_": (
            "Encoded Card Feature",
            "An encoded category derived from masked payment-card-related "
            "information."
        ),
        "card3_": (
            "Encoded Card Feature",
            "An encoded category derived from masked payment-card-related "
            "information."
        ),
        "card4_": (
            "Encoded Card Feature",
            "An encoded category derived from payment-card-related "
            "information."
        ),
        "card5_": (
            "Encoded Card Feature",
            "An encoded category derived from masked payment-card-related "
            "information."
        ),
        "card6_": (
            "Encoded Card Feature",
            "An encoded category derived from payment-card-related "
            "information."
        ),
        "P_emaildomain_": (
            "Encoded Purchaser Email Domain",
            "An encoded category derived from the purchaser email domain."
        ),
        "R_emaildomain_": (
            "Encoded Recipient Email Domain",
            "An encoded category derived from the recipient email domain."
        ),
        "DeviceType_": (
            "Encoded Device Type",
            "An encoded category derived from the device type."
        ),
        "DeviceInfo_": (
            "Encoded Device Information",
            "An encoded category derived from device information."
        ),
    }

    for prefix, description in encoded_prefixes.items():
        if feature.startswith(prefix):
            return description

    # Safe fallback for any field not covered by public documentation.
    return (
        feature,
        "No specific public description was provided for this field."
    )