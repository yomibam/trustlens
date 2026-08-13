from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import joblib
import pandas as pd
import streamlit as st

from src.prediction import predict_transactions
from src.feature_dictionary import get_feature_description

st.set_page_config(
    page_title="TrustLens",
    page_icon="🔍",
    layout="wide"
)

MODEL_PATH = ROOT_DIR / "models" / "lightgbm.pkl"
RESULTS_PATH = ROOT_DIR / "results" / "metrics" / "trustlens_model_comparison_table.csv"
SHAP_PLOT_PATH = ROOT_DIR / "results" / "figures" / "lightgbm_shap_summary_plot.png"
TOP_FEATURES_PATH = ROOT_DIR / "results" / "metrics" / "lightgbm_top_20_shap_features.csv"
MODEL_CONFIG_PATH = (
    ROOT_DIR
    / "results"
    / "metrics"
    / "model_configuration.csv"
)

DISPLAY_COLUMN_NAMES = {
    "TransactionID": "Transaction ID",
    "fraud_probability": "Fraud Probability",
    "predicted_isFraud": "Predicted Class",
    "risk_level": "Risk Category",
    "recommendation": "Recommended Action",
}

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

st.title("TrustLens")
st.caption(
    "An explainable fraud decision-support application for UK e-commerce transaction screening."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Analyse Transactions",
    "Transaction Investigation",
    "Model Insights"
])


with tab1:
    st.header("TrustLens Overview")

    st.write(
        "TrustLens is an explainable fraud decision-support application developed "
        "to support the screening and investigation of e-commerce transactions. "
        "It combines machine learning predictions with global explanation outputs "
        "and transaction-level assessment information to support model interpretation."
    )

    st.subheader("Selected Fraud-Classification Model")

    st.write(
        "The prototype uses a trained LightGBM binary classifier because it achieved "
        "the strongest overall predictive performance among the evaluated models."
    )

    if RESULTS_PATH.exists():
        st.subheader("Comparative Model Performance")
        st.dataframe(
            pd.read_csv(RESULTS_PATH),
            use_container_width=True
        )


with tab2:
    st.header("Analyse Transactions")

    st.write(
        "Upload preprocessed transaction records to generate fraud "
        "probabilities, predicted classes and decision-support guidance."
    )

    # ---------------------------------------------------------
    # Initialise saved analysis state
    # ---------------------------------------------------------

    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None

    if "analysis_verification" not in st.session_state:
        st.session_state.analysis_verification = None

    if "analysis_file_signature" not in st.session_state:
        st.session_state.analysis_file_signature = None

    # ---------------------------------------------------------
    # Upload transaction data
    # ---------------------------------------------------------

    upload_surface = st.container(border=True)
    upload_surface.subheader("1. Upload and Analyse")
    upload_surface.caption(
        "Select a compatible transaction file, inspect a concise preview and "
        "run the LightGBM fraud-classification model."
    )

    uploaded_file = upload_surface.file_uploader(
        "Upload a Preprocessed Transaction File",
        type=["csv"],
        key="analysis_upload",
        help=(
            "Upload transaction records containing the features expected "
            "by the trained LightGBM fraud-classification model."
        )
    )

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)

        # Reset old results when a different file is uploaded.
        current_file_signature = (
            uploaded_file.name,
            uploaded_file.size
        )

        if (
            st.session_state.analysis_file_signature
            != current_file_signature
        ):
            st.session_state.analysis_results = None
            st.session_state.analysis_verification = None
            st.session_state.analysis_file_signature = (
                current_file_signature
            )

        # Prevent completed TrustLens output files from being
        # uploaded as new model-input files.
        assessment_columns = {
            "fraud_probability",
            "predicted_isFraud",
            "risk_level",
            "recommendation"
        }

        if assessment_columns.intersection(input_df.columns):
            upload_surface.error(
                "This appears to be a completed TrustLens assessment file. "
                "Upload it in the Transaction Investigation tab instead."
            )

        else:
            # -------------------------------------------------
            # Display a concise input preview
            # -------------------------------------------------

            preview_candidates = [
                "TransactionID",
                "TransactionDT",
                "TransactionAmt",
                "ProductCD",
                "card4",
                "card6",
                "P_emaildomain",
                "DeviceType"
            ]

            input_preview_columns = [
                column
                for column in preview_candidates
                if column in input_df.columns
            ]

            # Use a limited fallback if the processed file does not
            # contain the original descriptive feature names.
            if not input_preview_columns:
                input_preview_columns = input_df.columns[:8].tolist()

            with upload_surface.expander(
                "Preview Uploaded Transactions",
                expanded=False
            ):
                st.dataframe(
                    input_df[input_preview_columns].head(10),
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(
                    f"{len(input_df):,} transactions and "
                    f"{len(input_df.columns):,} uploaded columns."
                )

            # -------------------------------------------------
            # Analyse uploaded transactions
            # -------------------------------------------------

            if upload_surface.button(
                "Analyse Transactions",
                type="primary",
                use_container_width=False
            ):
                try:
                    metadata_columns = ["TransactionID"]

                    model_input_check = input_df.drop(
                        columns=metadata_columns,
                        errors="ignore"
                    ).copy()

                    expected_features = list(model.feature_name_)

                    missing_features = [
                        feature
                        for feature in expected_features
                        if feature not in model_input_check.columns
                    ]

                    unexpected_features = [
                        feature
                        for feature in model_input_check.columns
                        if feature not in expected_features
                    ]

                    if missing_features:
                        upload_surface.error(
                            f"The uploaded file is missing "
                            f"{len(missing_features)} model features."
                        )

                        with upload_surface.expander("View Missing Features"):
                            st.write(missing_features)

                        st.stop()

                    if unexpected_features:
                        upload_surface.error(
                            f"The uploaded file contains "
                            f"{len(unexpected_features)} unexpected features."
                        )

                        with upload_surface.expander("View Unexpected Features"):
                            st.write(unexpected_features)

                        st.stop()

                    # Match the exact feature order used during training.
                    model_input_check = model_input_check[
                        expected_features
                    ]

                    feature_order_matches = (
                        list(model_input_check.columns)
                        == expected_features
                    )

                    verification_passed = (
                        model_input_check.shape[1]
                        == model.n_features_in_
                        and "TransactionID"
                        not in model_input_check.columns
                        and feature_order_matches
                    )

                    if not verification_passed:
                        upload_surface.error(
                            "Model input verification failed. The uploaded "
                            "transaction data does not match the trained "
                            "LightGBM model."
                        )
                        st.stop()

                    predictions = predict_transactions(
                        model,
                        input_df
                    )

                    # Save the results so filters can rerun without
                    # removing the completed analysis.
                    st.session_state.analysis_results = (
                        predictions.copy()
                    )

                    st.session_state.analysis_verification = {
                        "expected_features": model.n_features_in_,
                        "received_features": (
                            model_input_check.shape[1]
                        ),
                        "transaction_id_included": (
                            "TransactionID"
                            in model_input_check.columns
                        ),
                        "feature_order_matches": (
                            feature_order_matches
                        )
                    }

                    upload_surface.success(
                        "Transaction analysis completed successfully."
                    )

                except ValueError as error:
                    upload_surface.error(str(error))

            # -------------------------------------------------
            # Display saved analysis results
            # -------------------------------------------------

            if st.session_state.analysis_results is not None:
                predictions = (
                    st.session_state.analysis_results.copy()
                )

                st.divider()

                # ---------------------------------------------
                # Overall assessment summary
                # ---------------------------------------------

                summary_surface = st.container(border=True)
                summary_surface.subheader("2. Assessment Summary")

                summary_col1, summary_col2, summary_col3 = (
                    summary_surface.columns(3)
                )

                summary_col1.metric(
                    "Transactions Analysed",
                    f"{len(predictions):,}"
                )

                summary_col2.metric(
                    "Transactions Predicted as Fraud",
                    f"{int(predictions['predicted_isFraud'].sum()):,}"
                )

                summary_col3.metric(
                    "Mean Fraud Probability",
                    (
                        f"{predictions['fraud_probability'].mean():.2%}"
                    )
                )

                # ---------------------------------------------
                # Collapsed technical verification
                # ---------------------------------------------

                verification = (
                    st.session_state.analysis_verification
                )

                if verification is not None:
                    with st.expander(
                        "Technical Verification",
                        expanded=False
                    ):
                        (
                            verification_col1,
                            verification_col2,
                            verification_col3
                        ) = st.columns(3)

                        verification_col1.metric(
                            "Model Expected Features",
                            verification["expected_features"]
                        )

                        verification_col2.metric(
                            "Features Received",
                            verification["received_features"]
                        )

                        verification_col3.metric(
                            "Transaction ID Included",
                            (
                                "Yes"
                                if verification[
                                    "transaction_id_included"
                                ]
                                else "No"
                            )
                        )

                        st.success(
                            "Verification passed. The feature count, "
                            "feature names and feature order matched the "
                            "trained LightGBM model. Transaction ID was "
                            "excluded from the predictive input."
                        )

                # ---------------------------------------------
                # Transaction assessment filters
                # ---------------------------------------------

                filter_surface = st.container(border=True)
                filter_surface.subheader("3. Filter Transactions")

                filter_surface.caption(
                    "Use the controls to focus on transactions requiring "
                    "particular levels of review."
                )

                filter_col1, filter_col2 = filter_surface.columns(2)

                with filter_col1:
                    transaction_search = st.text_input(
                        "Search Transaction ID",
                        placeholder="Enter a complete or partial ID",
                        key="transaction_id_filter"
                    )

                    minimum_probability = st.slider(
                        "Minimum Fraud Probability",
                        min_value=0,
                        max_value=100,
                        value=0,
                        step=1,
                        format="%d%%",
                        key="minimum_probability_filter"
                    )

                with filter_col2:
                    predicted_class_filter = st.selectbox(
                        "Predicted Class",
                        options=[
                            "All Classes",
                            "Predicted as Fraud",
                            "Predicted as Non-Fraud"
                        ],
                        key="predicted_class_filter"
                    )

                    available_risk_categories = sorted(
                        predictions["risk_level"]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )

                    selected_risk_categories = st.multiselect(
                        "Risk Category",
                        options=available_risk_categories,
                        default=available_risk_categories,
                        key="risk_category_filter"
                    )

                rows_to_display = filter_surface.selectbox(
                    "Rows to Display",
                    options=[10, 25, 50, 100, "All"],
                    index=2,
                    key="rows_to_display_filter"
                )

                # ---------------------------------------------
                # Apply filters
                # ---------------------------------------------

                filtered_results = predictions.copy()

                if (
                    transaction_search.strip()
                    and "TransactionID"
                    in filtered_results.columns
                ):
                    filtered_results = filtered_results[
                        filtered_results["TransactionID"]
                        .astype(str)
                        .str.contains(
                            transaction_search.strip(),
                            case=False,
                            na=False
                        )
                    ]

                if (
                    predicted_class_filter
                    == "Predicted as Fraud"
                ):
                    filtered_results = filtered_results[
                        filtered_results["predicted_isFraud"] == 1
                    ]

                elif (
                    predicted_class_filter
                    == "Predicted as Non-Fraud"
                ):
                    filtered_results = filtered_results[
                        filtered_results["predicted_isFraud"] == 0
                    ]

                if selected_risk_categories:
                    filtered_results = filtered_results[
                        filtered_results["risk_level"]
                        .astype(str)
                        .isin(selected_risk_categories)
                    ]
                else:
                    # An empty selection means no risk categories
                    # should be displayed.
                    filtered_results = filtered_results.iloc[0:0]

                filtered_results = filtered_results[
                    filtered_results["fraud_probability"]
                    >= minimum_probability / 100
                ]

                # Highest-risk transactions appear first.
                filtered_results = filtered_results.sort_values(
                    by="fraud_probability",
                    ascending=False
                )

                results_surface = st.container(border=True)
                results_surface.subheader("4. Review and Export Results")

                results_surface.caption(
                    f"Showing {len(filtered_results):,} of "
                    f"{len(predictions):,} assessed transactions."
                )

                # ---------------------------------------------
                # Prepare the user-facing table
                # ---------------------------------------------

                table_columns = [
                    column
                    for column in [
                        "TransactionID",
                        "fraud_probability",
                        "predicted_isFraud",
                        "risk_level",
                        "recommendation"
                    ]
                    if column in filtered_results.columns
                ]

                if rows_to_display == "All":
                    displayed_results = (
                        filtered_results[table_columns].copy()
                    )
                else:
                    displayed_results = (
                        filtered_results[table_columns]
                        .head(rows_to_display)
                        .copy()
                    )

                if (
                    "predicted_isFraud"
                    in displayed_results.columns
                ):
                    displayed_results["predicted_isFraud"] = (
                        displayed_results["predicted_isFraud"]
                        .map({
                            1: "Predicted as Fraud",
                            0: "Predicted as Non-Fraud"
                        })
                    )

                displayed_results = displayed_results.rename(
                    columns={
                        "TransactionID": "Transaction ID",
                        "fraud_probability": "Fraud Probability",
                        "predicted_isFraud": "Predicted Class",
                        "risk_level": "Risk Category",
                        "recommendation": "Recommended Action"
                    }
                )

                if (
                    "Fraud Probability"
                    in displayed_results.columns
                ):
                    displayed_results["Fraud Probability"] = (
                        displayed_results["Fraud Probability"]
                        * 100
                    )

                if displayed_results.empty:
                    results_surface.info(
                        "No transactions match the selected filters."
                    )

                else:
                    results_surface.dataframe(
                        displayed_results,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Transaction ID": (
                                st.column_config.TextColumn(
                                    "Transaction ID"
                                )
                            ),
                            "Fraud Probability": (
                                st.column_config.ProgressColumn(
                                    "Fraud Probability",
                                    help=(
                                        "Probability assigned by the "
                                        "LightGBM fraud-classification "
                                        "model."
                                    ),
                                    min_value=0.0,
                                    max_value=100.0,
                                    format="%.2f%%"
                                )
                            ),
                            "Predicted Class": (
                                st.column_config.TextColumn(
                                    "Predicted Class",
                                    help=(
                                        "Classification produced using "
                                        "the configured model threshold."
                                    )
                                )
                            ),
                            "Risk Category": (
                                st.column_config.TextColumn(
                                    "Risk Category",
                                    help=(
                                        "Decision-support category "
                                        "assigned by TrustLens."
                                    )
                                )
                            ),
                            "Recommended Action": (
                                st.column_config.TextColumn(
                                    "Recommended Action",
                                    help=(
                                        "Operational guidance assigned "
                                        "from the configured risk rules."
                                    )
                                )
                            )
                        }
                    )

                # ---------------------------------------------
                # Download results
                # ---------------------------------------------

                download_col1, download_col2 = results_surface.columns(2)

                complete_csv = predictions.to_csv(
                    index=False
                ).encode("utf-8")

                filtered_csv = filtered_results.to_csv(
                    index=False
                ).encode("utf-8")

                with download_col1:
                    st.download_button(
                        label="Download Complete Results",
                        data=complete_csv,
                        file_name=(
                            "trustlens_assessment_results.csv"
                        ),
                        mime="text/csv",
                        use_container_width=True
                    )

                with download_col2:
                    st.download_button(
                        label="Download Filtered Results",
                        data=filtered_csv,
                        file_name=(
                            "trustlens_filtered_results.csv"
                        ),
                        mime="text/csv",
                        use_container_width=True
                    )


with tab3:
    st.header("Transaction Investigation")

    st.write(
        "Upload exported TrustLens assessment results, filter the assessed "
        "transactions and examine an individual transaction in greater detail."
    )

    # ---------------------------------------------------------
    # Upload completed TrustLens assessment results
    # ---------------------------------------------------------

    with st.container(border=True):
        st.subheader("1. Upload Assessment Results")

        investigation_file = st.file_uploader(
            "Upload TrustLens Assessment Results",
            type=["csv"],
            key="investigation_upload",
            help=(
                "Upload the CSV exported from the Analyse Transactions "
                "section."
            )
        )

        if investigation_file is None:
            st.info(
                "Upload a completed TrustLens assessment file to begin "
                "transaction investigation."
            )

    if investigation_file is not None:
        investigation_df = pd.read_csv(investigation_file)

        required_columns = {
            "fraud_probability",
            "predicted_isFraud",
            "risk_level",
            "recommendation"
        }

        if not required_columns.issubset(
            investigation_df.columns
        ):
            missing_columns = sorted(
                required_columns
                - set(investigation_df.columns)
            )

            st.error(
                "This file does not contain the required TrustLens "
                "assessment fields."
            )

            with st.expander("View Missing Assessment Fields"):
                st.write(missing_columns)

        else:
            # Ensure fields have predictable data types.
            investigation_df["fraud_probability"] = (
                pd.to_numeric(
                    investigation_df["fraud_probability"],
                    errors="coerce"
                )
            )

            investigation_df["predicted_isFraud"] = (
                pd.to_numeric(
                    investigation_df["predicted_isFraud"],
                    errors="coerce"
                )
            )

            # Highest-probability transactions appear first.
            investigation_df = investigation_df.sort_values(
                by="fraud_probability",
                ascending=False
            )

            # -------------------------------------------------
            # Create the list-detail layout
            # -------------------------------------------------

            selection_pane, detail_pane = st.columns(
                [1, 2],
                gap="large"
            )

            # =================================================
            # LEFT PANE: Filters and transaction selection
            # =================================================

            with selection_pane:
                with st.container(border=True):
                    st.subheader("2. Find a Transaction")

                    transaction_search = st.text_input(
                        "Search Transaction ID",
                        placeholder="Enter a complete or partial ID",
                        key="investigation_transaction_search"
                    )

                    predicted_class_filter = st.selectbox(
                        "Predicted Class",
                        options=[
                            "All Classes",
                            "Predicted as Fraud",
                            "Predicted as Non-Fraud"
                        ],
                        key="investigation_class_filter"
                    )

                    risk_order = [
                        "Critical",
                        "High",
                        "Medium",
                        "Moderate",
                        "Low"
                    ]

                    existing_risk_categories = (
                        investigation_df["risk_level"]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )

                    available_risk_categories = [
                        risk
                        for risk in risk_order
                        if risk in existing_risk_categories
                    ]

                    available_risk_categories.extend(
                        sorted(
                            risk
                            for risk in existing_risk_categories
                            if risk not in risk_order
                        )
                    )

                    selected_risk_categories = st.multiselect(
                        "Risk Category",
                        options=available_risk_categories,
                        default=available_risk_categories,
                        key="investigation_risk_filter"
                    )

                    minimum_probability = st.slider(
                        "Minimum Fraud Probability",
                        min_value=0,
                        max_value=100,
                        value=0,
                        step=1,
                        format="%d%%",
                        key="investigation_probability_filter"
                    )

                    # -----------------------------------------
                    # Apply investigation filters
                    # -----------------------------------------

                    filtered_investigation_df = (
                        investigation_df.copy()
                    )

                    if (
                        transaction_search.strip()
                        and "TransactionID"
                        in filtered_investigation_df.columns
                    ):
                        filtered_investigation_df = (
                            filtered_investigation_df[
                                filtered_investigation_df[
                                    "TransactionID"
                                ]
                                .astype(str)
                                .str.contains(
                                    transaction_search.strip(),
                                    case=False,
                                    na=False
                                )
                            ]
                        )

                    if (
                        predicted_class_filter
                        == "Predicted as Fraud"
                    ):
                        filtered_investigation_df = (
                            filtered_investigation_df[
                                filtered_investigation_df[
                                    "predicted_isFraud"
                                ] == 1
                            ]
                        )

                    elif (
                        predicted_class_filter
                        == "Predicted as Non-Fraud"
                    ):
                        filtered_investigation_df = (
                            filtered_investigation_df[
                                filtered_investigation_df[
                                    "predicted_isFraud"
                                ] == 0
                            ]
                        )

                    if selected_risk_categories:
                        filtered_investigation_df = (
                            filtered_investigation_df[
                                filtered_investigation_df[
                                    "risk_level"
                                ]
                                .astype(str)
                                .isin(
                                    selected_risk_categories
                                )
                            ]
                        )
                    else:
                        filtered_investigation_df = (
                            filtered_investigation_df.iloc[0:0]
                        )

                    filtered_investigation_df = (
                        filtered_investigation_df[
                            filtered_investigation_df[
                                "fraud_probability"
                            ]
                            >= minimum_probability / 100
                        ]
                    )

                    st.caption(
                        f"{len(filtered_investigation_df):,} of "
                        f"{len(investigation_df):,} transactions match "
                        f"the selected filters."
                    )

                    if filtered_investigation_df.empty:
                        st.warning(
                            "No transactions match the selected filters."
                        )

                        selected_index = None

                    else:
                        def format_transaction_option(
                            row_index
                        ) -> str:
                            row = filtered_investigation_df.loc[
                                row_index
                            ]

                            if (
                                "TransactionID"
                                in row.index
                                and pd.notna(
                                    row["TransactionID"]
                                )
                            ):
                                transaction_id = row[
                                    "TransactionID"
                                ]

                                try:
                                    transaction_id = int(
                                        float(transaction_id)
                                    )
                                except (
                                    TypeError,
                                    ValueError
                                ):
                                    pass

                                identifier = (
                                    f"ID {transaction_id}"
                                )

                            else:
                                identifier = (
                                    f"Row {row_index}"
                                )

                            probability_label = (
                                f"{row['fraud_probability']:.2%}"
                            )

                            risk_label = str(
                                row["risk_level"]
                            )

                            return (
                                f"{identifier} · "
                                f"{probability_label} · "
                                f"{risk_label}"
                            )

                        selected_index = st.selectbox(
                            "Select Transaction",
                            options=(
                                filtered_investigation_df
                                .index
                                .tolist()
                            ),
                            format_func=(
                                format_transaction_option
                            ),
                            key="investigation_transaction_selector"
                        )

            # =================================================
            # RIGHT PANE: Selected transaction details
            # =================================================

            with detail_pane:
                if selected_index is None:
                    with st.container(border=True):
                        st.subheader(
                            "Transaction Assessment"
                        )

                        st.info(
                            "Adjust the filters to select a "
                            "transaction for investigation."
                        )

                else:
                    selected_row = (
                        filtered_investigation_df
                        .loc[selected_index]
                        .copy()
                    )

                    integer_columns = [
                        "TransactionID",
                        "TransactionDT",
                        "card1",
                        "card2",
                        "card3",
                        "card5",
                        "addr1",
                        "addr2"
                    ]

                    for column in integer_columns:
                        if (
                            column in selected_row.index
                            and pd.notna(
                                selected_row[column]
                            )
                        ):
                            try:
                                selected_row[column] = int(
                                    float(
                                        selected_row[column]
                                    )
                                )
                            except (
                                TypeError,
                                ValueError
                            ):
                                pass

                    fraud_probability = float(
                        selected_row[
                            "fraud_probability"
                        ]
                    )

                    predicted_value = int(
                        float(
                            selected_row[
                                "predicted_isFraud"
                            ]
                        )
                    )

                    predicted_class = (
                        "Predicted as Fraud"
                        if predicted_value == 1
                        else "Predicted as Non-Fraud"
                    )

                    risk_category = str(
                        selected_row["risk_level"]
                    )

                    recommended_action = str(
                        selected_row["recommendation"]
                    )

                    # -----------------------------------------
                    # Primary transaction assessment
                    # -----------------------------------------

                    with st.container(border=True):
                        if (
                            "TransactionID"
                            in selected_row.index
                            and pd.notna(
                                selected_row["TransactionID"]
                            )
                        ):
                            st.subheader(
                                "Transaction "
                                f"{selected_row['TransactionID']}"
                            )
                        else:
                            st.subheader(
                                "Selected Transaction"
                            )

                        metric_col1, metric_col2, metric_col3 = (
                            st.columns(3)
                        )

                        metric_col1.metric(
                            "Fraud Probability",
                            f"{fraud_probability:.2%}"
                        )

                        metric_col2.metric(
                            "Predicted Class",
                            predicted_class
                        )

                        metric_col3.metric(
                            "Risk Category",
                            risk_category
                        )

                        st.markdown(
                            "**Recommended Action**"
                        )

                        if risk_category.lower() in {
                            "critical",
                            "high"
                        }:
                            st.warning(recommended_action)

                        else:
                            st.info(recommended_action)

                    # -----------------------------------------
                    # Decision-support interpretation
                    # -----------------------------------------

                    with st.container(border=True):
                        st.subheader(
                            "Decision-Support Summary"
                        )

                        if fraud_probability >= 0.85:
                            summary = (
                                "TrustLens categorised this transaction "
                                "as Critical Risk based on the fraud "
                                "probability produced by the LightGBM "
                                "model. Immediate escalation or blocking "
                                "is recommended under the configured "
                                "prototype decision rules."
                            )

                        elif fraud_probability >= 0.60:
                            summary = (
                                "TrustLens categorised this transaction "
                                "as High Risk. Manual review is "
                                "recommended before the transaction is "
                                "approved."
                            )

                        elif fraud_probability >= 0.30:
                            summary = (
                                "TrustLens categorised this transaction "
                                "as Medium Risk. Additional verification "
                                "may be appropriate before a final "
                                "decision is made."
                            )

                        else:
                            summary = (
                                "TrustLens categorised this transaction "
                                "as Low Risk based on the current model "
                                "probability. This assessment should not "
                                "be treated as confirmation that the "
                                "transaction is legitimate."
                            )

                        st.info(summary)

                        st.caption(
                            "The risk category and recommended action "
                            "are produced by TrustLens decision rules. "
                            "They are not confirmation that fraud "
                            "occurred."
                        )

                    # -----------------------------------------
                    # Important transaction information
                    # -----------------------------------------

                    with st.container(border=True):
                        st.subheader(
                            "Key Transaction Information"
                        )

                        important_features = [
                            "TransactionID",
                            "TransactionDT",
                            "TransactionAmt",
                            "ProductCD",
                            "card4",
                            "card6",
                            "P_emaildomain",
                            "R_emaildomain",
                            "DeviceType",
                            "addr1",
                            "addr2"
                        ]

                        key_information_rows = []

                        for feature in important_features:
                            if feature in selected_row.index:
                                (
                                    display_name,
                                    description
                                ) = get_feature_description(
                                    feature
                                )

                                value = selected_row[feature]

                                if pd.isna(value):
                                    value = "Not available"

                                key_information_rows.append({
                                    "Feature": display_name,
                                    "Value": value,
                                    "Description": description
                                })

                        if key_information_rows:
                            st.dataframe(
                                pd.DataFrame(
                                    key_information_rows
                                ),
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "Feature": (
                                        st.column_config.TextColumn(
                                            "Feature",
                                            width="medium"
                                        )
                                    ),
                                    "Value": (
                                        st.column_config.TextColumn(
                                            "Value",
                                            width="medium"
                                        )
                                    ),
                                    "Description": (
                                        st.column_config.TextColumn(
                                            "Public Description",
                                            width="large"
                                        )
                                    )
                                }
                            )

                        else:
                            st.info(
                                "No key descriptive transaction fields "
                                "were available in this assessment file."
                            )

                    # -----------------------------------------
                    # Filtered advanced technical fields
                    # -----------------------------------------

                    with st.expander(
                        "Advanced Technical Details",
                        expanded=False
                    ):
                        st.caption(
                            "Use the controls below to limit the technical "
                            "fields displayed."
                        )

                        technical_filter_col1, technical_filter_col2 = (
                            st.columns(2)
                        )

                        with technical_filter_col1:
                            technical_field_group = st.selectbox(
                                "Field Group",
                                options=[
                                    "Model Input Fields",
                                    "TrustLens Output Fields",
                                    "All Fields"
                                ],
                                key="technical_field_group"
                            )

                        with technical_filter_col2:
                            technical_field_search = st.text_input(
                                "Search Field",
                                placeholder=(
                                    "For example: TransactionAmt or V70"
                                ),
                                key="technical_field_search"
                            )

                        technical_rows_limit = st.selectbox(
                            "Rows to Display",
                            options=[10, 25, 50, 100, "All"],
                            index=1,
                            key="technical_rows_limit"
                        )

                        trustlens_output_fields = {
                            "fraud_probability",
                            "predicted_isFraud",
                            "risk_level",
                            "recommendation"
                        }

                        advanced_rows = []

                        for feature, value in selected_row.items():
                            if (
                                technical_field_group
                                == "Model Input Fields"
                                and feature
                                in trustlens_output_fields
                            ):
                                continue

                            if (
                                technical_field_group
                                == "TrustLens Output Fields"
                                and feature
                                not in trustlens_output_fields
                            ):
                                continue

                            (
                                display_name,
                                description
                            ) = get_feature_description(
                                feature
                            )

                            if (
                                technical_field_search.strip()
                                and technical_field_search
                                .strip()
                                .lower()
                                not in feature.lower()
                                and technical_field_search
                                .strip()
                                .lower()
                                not in display_name.lower()
                            ):
                                continue

                            if pd.isna(value):
                                value = "Not available"

                            advanced_rows.append({
                                "Feature": display_name,
                                "Original Feature": feature,
                                "Value": value,
                                "Description": description
                            })

                        advanced_details_df = pd.DataFrame(
                            advanced_rows
                        )

                        if technical_rows_limit != "All":
                            advanced_details_df = (
                                advanced_details_df.head(
                                    technical_rows_limit
                                )
                            )

                        if advanced_details_df.empty:
                            st.info(
                                "No technical fields match the "
                                "selected filters."
                            )

                        else:
                            st.dataframe(
                                advanced_details_df,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "Feature": (
                                        st.column_config.TextColumn(
                                            "Feature",
                                            width="medium"
                                        )
                                    ),
                                    "Original Feature": (
                                        st.column_config.TextColumn(
                                            "Original Feature",
                                            width="medium"
                                        )
                                    ),
                                    "Value": (
                                        st.column_config.TextColumn(
                                            "Value",
                                            width="medium"
                                        )
                                    ),
                                    "Description": (
                                        st.column_config.TextColumn(
                                            "Public Description",
                                            width="large"
                                        )
                                    )
                                }
                            )

with tab4:
    st.header("Model Insights")

    st.write(
        "Review the selected fraud-classification model, compare predictive "
        "and operational performance, and inspect the global SHAP explanation."
    )

    if not RESULTS_PATH.exists():
        st.warning(
            "The comparative model-results file could not be found. "
            "Confirm that the results directory is available."
        )

    else:
        model_results = pd.read_csv(RESULTS_PATH)

        selected_model_rows = model_results[
            model_results["model"]
            .astype(str)
            .str.casefold()
            == "lightgbm"
        ]

        # -----------------------------------------------------
        # Selected model summary
        # -----------------------------------------------------

        with st.container(border=True):
            st.subheader("1. Selected Fraud-Classification Model")

            if selected_model_rows.empty:
                st.warning(
                    "LightGBM was not found in the comparative-results file."
                )

            else:
                selected_model = selected_model_rows.iloc[0]

                metric_col1, metric_col2, metric_col3, metric_col4 = (
                    st.columns(4)
                )

                metric_col1.metric(
                    "ROC-AUC",
                    f"{selected_model['roc_auc']:.3f}"
                )

                metric_col2.metric(
                    "PR-AUC",
                    f"{selected_model['pr_auc']:.3f}"
                )

                metric_col3.metric(
                    "Recall",
                    f"{selected_model['recall']:.2%}"
                )

                metric_col4.metric(
                    "MCC",
                    f"{selected_model['mcc']:.3f}"
                )

                st.write(
                    "LightGBM was selected because it achieved the strongest "
                    "overall predictive performance among the evaluated models, "
                    "including the highest PR-AUC, recall, F1 score and MCC."
                )

                st.caption(
                    f"The trained classifier expects "
                    f"{model.n_features_in_:,} predictive features. "
                    "TrustLens converts its fraud probability into a predicted "
                    "class, risk category and recommended action."
                )

                if MODEL_CONFIG_PATH.exists():
                    with st.expander(
                        "LightGBM Model Configuration",
                        expanded=False
                    ):
                        model_configuration = pd.read_csv(
                            MODEL_CONFIG_PATH
                        )

                        model_column = (
                            "Model"
                            if "Model" in model_configuration.columns
                            else "model"
                        )

                        lightgbm_configuration = model_configuration[
                            model_configuration[model_column]
                            .astype(str)
                            .str.casefold()
                            == "lightgbm"
                        ]

                        if lightgbm_configuration.empty:
                            st.info(
                                "No LightGBM configuration row was found."
                            )

                        else:
                            configuration_row = (
                                lightgbm_configuration.iloc[0]
                            )

                            configuration_fields = {
                                "n_estimators": "Number of Trees",
                                "learning_rate": "Learning Rate",
                                "num_leaves": "Number of Leaves",
                                "max_depth": "Maximum Depth",
                                "scale_pos_weight": "Class Weight Ratio",
                                "random_state": "Random State",
                                "n_jobs": "Processing Jobs"
                            }

                            configuration_rows = []

                            for field, display_name in (
                                configuration_fields.items()
                            ):
                                if (
                                    field in configuration_row.index
                                    and pd.notna(configuration_row[field])
                                ):
                                    value = configuration_row[field]

                                    if isinstance(value, float):
                                        if value.is_integer():
                                            value = int(value)
                                        else:
                                            value = round(value, 6)

                                    configuration_rows.append({
                                        "Setting": display_name,
                                        "Value": value
                                    })

                            st.dataframe(
                                pd.DataFrame(configuration_rows),
                                use_container_width=True,
                                hide_index=True
                            )

        # -----------------------------------------------------
        # Comparative predictive performance
        # -----------------------------------------------------

        with st.container(border=True):
            st.subheader("2. Comparative Predictive Performance")

            st.caption(
                "Select a metric group to limit the information displayed."
            )

            metric_group = st.selectbox(
                "Metric Group",
                options=[
                    "Fraud-Detection Metrics",
                    "Discrimination Metrics",
                    "All Predictive Metrics"
                ],
                key="model_metric_group"
            )

            predictive_column_groups = {
                "Fraud-Detection Metrics": [
                    "model",
                    "precision",
                    "recall",
                    "f1_score",
                    "pr_auc",
                    "mcc"
                ],
                "Discrimination Metrics": [
                    "model",
                    "accuracy",
                    "roc_auc",
                    "pr_auc"
                ],
                "All Predictive Metrics": [
                    "model",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1_score",
                    "roc_auc",
                    "pr_auc",
                    "mcc"
                ]
            }

            selected_columns = [
                column
                for column in predictive_column_groups[metric_group]
                if column in model_results.columns
            ]

            predictive_table = model_results[
                selected_columns
            ].copy()

            predictive_table.insert(
                1,
                "selected_model",
                predictive_table["model"]
                .astype(str)
                .str.casefold()
                .eq("lightgbm")
                .map({True: "Selected", False: ""})
            )

            metric_columns = [
                column
                for column in predictive_table.columns
                if column not in {"model", "selected_model"}
            ]

            for column in metric_columns:
                predictive_table[column] = (
                    pd.to_numeric(
                        predictive_table[column],
                        errors="coerce"
                    ).round(3)
                )

            if "pr_auc" in predictive_table.columns:
                predictive_table = predictive_table.sort_values(
                    by="pr_auc",
                    ascending=False
                )

            predictive_table = predictive_table.rename(
                columns={
                    "model": "Model",
                    "selected_model": "Status",
                    "accuracy": "Accuracy",
                    "precision": "Precision",
                    "recall": "Recall",
                    "f1_score": "F1 Score",
                    "roc_auc": "ROC-AUC",
                    "pr_auc": "PR-AUC",
                    "mcc": "MCC"
                }
            )

            st.dataframe(
                predictive_table,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "PR-AUC, recall, F1 score and MCC are particularly useful "
                "for interpreting performance on the imbalanced fraud dataset."
            )

        # -----------------------------------------------------
        # Operational performance
        # -----------------------------------------------------

        with st.container(border=True):
            st.subheader("3. Operational Performance")

            operational_columns = [
                column
                for column in [
                    "model",
                    "training_time_seconds",
                    "prediction_time_seconds",
                    "interpretation_method",
                    "shap_explanation_time_500_samples"
                ]
                if column in model_results.columns
            ]

            operational_table = model_results[
                operational_columns
            ].copy()

            numeric_operational_columns = [
                "training_time_seconds",
                "prediction_time_seconds",
                "shap_explanation_time_500_samples"
            ]

            for column in numeric_operational_columns:
                if column in operational_table.columns:
                    operational_table[column] = (
                        pd.to_numeric(
                            operational_table[column],
                            errors="coerce"
                        ).round(3)
                    )

            operational_table = operational_table.rename(
                columns={
                    "model": "Model",
                    "training_time_seconds": "Training Time (s)",
                    "prediction_time_seconds": "Prediction Time (s)",
                    "interpretation_method": "Interpretation Method",
                    "shap_explanation_time_500_samples": (
                        "Explanation Time — 500 Samples (s)"
                    )
                }
            )

            st.dataframe(
                operational_table,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "These values are experimental timings from the recorded "
                "evaluation runs. They should not be interpreted as "
                "per-transaction production latency. Logistic Regression "
                "used model coefficients rather than SHAP."
            )

    # ---------------------------------------------------------
    # Global SHAP explanation
    # ---------------------------------------------------------

    with st.container(border=True):
        st.subheader("4. Global SHAP Explanation")

        st.write(
            "The global SHAP analysis summarises how strongly each feature "
            "influenced LightGBM predictions across the explanation sample. "
            "Larger mean absolute SHAP values indicate greater overall model "
            "influence, not proof that a feature caused fraud."
        )

        if SHAP_PLOT_PATH.exists():
            st.image(
                str(SHAP_PLOT_PATH),
                caption=(
                    "Global SHAP summary for the selected LightGBM model"
                ),
                use_container_width=True
            )

        else:
            st.info(
                "The global SHAP summary plot could not be found."
            )

        if TOP_FEATURES_PATH.exists():
            top_features = pd.read_csv(TOP_FEATURES_PATH).rename(
                columns={
                    "mean_absolute_shap_value": "mean_abs_shap"
                }
            )

            shap_filter_col1, shap_filter_col2 = st.columns(2)

            with shap_filter_col1:
                feature_search = st.text_input(
                    "Search Feature",
                    placeholder="For example: TransactionAmt or C13",
                    key="global_shap_feature_search"
                )

            with shap_filter_col2:
                maximum_features = min(
                    20,
                    len(top_features)
                )

                top_n_features = st.slider(
                    "Features to Display",
                    min_value=5,
                    max_value=maximum_features,
                    value=min(10, maximum_features),
                    step=1,
                    key="global_shap_top_n"
                )

            shap_rows = []

            for _, feature_row in top_features.iterrows():
                feature = str(feature_row["feature"])
                display_name, description = (
                    get_feature_description(feature)
                )

                if (
                    feature_search.strip()
                    and feature_search.strip().lower()
                    not in feature.lower()
                    and feature_search.strip().lower()
                    not in display_name.lower()
                ):
                    continue

                shap_rows.append({
                    "Feature": display_name,
                    "Original Feature": feature,
                    "Mean Absolute SHAP Value": round(
                        float(feature_row["mean_abs_shap"]),
                        6
                    ),
                    "Public Description": description
                })

            shap_table = pd.DataFrame(
                shap_rows
            ).head(top_n_features)

            if shap_table.empty:
                st.info(
                    "No SHAP features match the search term."
                )

            else:
                st.dataframe(
                    shap_table,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Feature": st.column_config.TextColumn(
                            "Feature",
                            width="medium"
                        ),
                        "Original Feature": (
                            st.column_config.TextColumn(
                                "Original Feature",
                                width="small"
                            )
                        ),
                        "Mean Absolute SHAP Value": (
                            st.column_config.NumberColumn(
                                "Mean Absolute SHAP Value",
                                format="%.6f"
                            )
                        ),
                        "Public Description": (
                            st.column_config.TextColumn(
                                "Public Description",
                                width="large"
                            )
                        )
                    }
                )

        else:
            st.info(
                "The global SHAP feature-importance table could not be found."
            )

    # ---------------------------------------------------------
    # Model scope, disclosure and limitations
    # ---------------------------------------------------------

    with st.container(border=True):
        st.subheader("5. Model Scope and Limitations")

        st.info(
            "The IEEE-CIS dataset is publicly accessible, but the original "
            "provider did not disclose the complete semantic definitions of "
            "many engineered variables. Features such as C13, V70 and V258 "
            "therefore remain anonymised. TrustLens reports their statistical "
            "influence without assigning unsupported business meanings."
        )

        st.warning(
            "SHAP explains how features influenced the model output; it does "
            "not establish why fraud occurred. TrustLens is a research "
            "prototype and its risk categories and recommended actions are "
            "decision-support rules rather than confirmed fraud outcomes."
        )

        with st.expander("Additional Prototype Limitations"):
            st.markdown(
                """
                - The model was evaluated using a historical public dataset rather than live UK merchant transactions.
                - The recorded timings do not constitute a production real-time latency test.
                - Feature anonymisation limits business-level interpretation of several SHAP results.
                - The interface has not undergone a formal usability study with professional fraud analysts.
                - Model performance may change when transaction behaviour or fraud patterns drift over time.
                """
            )
