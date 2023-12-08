import numpy as np
import pandas as pd
import streamlit as st
import pickle

# Configuring the Streamlit page with a title, icon, and an about section in the menu
st.set_page_config(
    page_title="Prediction Tool",
    page_icon="🔮",
    menu_items={
        "About": """This web application is a project that aims to assist AllLife Bank in identifying potential customers 
        likely to accept personal loan offers. It is based on a dataset of customer demographics and financial behavior."""
    },
)

############################################################################################

# Function to load and cache the data to improve performance
file = "Loan_Modelling.csv"


@st.cache_data
def load_data(file):
    cached_data = pd.read_csv(file)
    return cached_data


cached_data = load_data(file)
df = cached_data.copy()

############################################################################################

# Loading pre-trained models and a scaler from disk using pickle
with open("models/roc_auc_model.pkl", "rb") as file:
    roc_auc_model = pickle.load(file)

with open("models/f1weight_model.pkl", "rb") as file:
    f1weight_model = pickle.load(file)

with open("models/recall_model.pkl", "rb") as file:
    recall_model = pickle.load(file)

with open("models/precision_model.pkl", "rb") as file:
    precision_model = pickle.load(file)

with open("models/StandardScaler.pkl", "rb") as file:
    StandardScaler = pickle.load(file)

############################################################################################

# Streamlit UI elements for displaying the title, an image, and a subheader for input
st.title("🏦 Customer Loan Prediction Tool")
st.image("images/hand.jpg")

st.subheader("Input Customer Profile")

############################################################################################


# Creating columns in the Streamlit app for organizing input fields
co1, co2, co3 = st.columns(3)

# Input fields for customer profile data, grouped into three columns
with co1:
    # Input fields for income, experience, and CD account status
    income = st.number_input("Annual Income ($)", value=115000, min_value=0)
    experience = st.number_input(
        "Work Experience (Years)", value=5, min_value=0, max_value=50
    )
    cd_account = st.selectbox(
        "Certificate of Deposit Account", ("Yes", "No"), index=1)

with co2:
    # Input fields for credit use, family size, securities account, and region
    avg_credit_use = st.number_input(
        "Average Credit Card Use ($)", value=1500, min_value=0
    )
    family_size = st.number_input(
        "Family Size", value=2, min_value=1, max_value=10)
    securities_account = st.selectbox(
        "Securities Account", ("Yes", "No"), index=1)
    region = st.selectbox(
        "Region",
        ("Central", "Los Angeles", "Southern", "Superior", "Bay Area"),
        index=1,
    )

with co3:
    # Input fields for mortgage value, education level, online banking, and credit card use
    mortgage = st.number_input(
        "House Mortgage Value ($)", min_value=0, value=189000)
    education_level = st.selectbox(
        "Education Level", ("Undergrad", "Graduate", "Professional"), index=2
    )
    online_banking = st.selectbox("Online Banking", ("Yes", "No"))
    credit_card = st.selectbox("Credit Card User", ("Yes", "No"), index=1)

############################################################################################

# Data processing and transformation for model input
# Mapping categorical inputs to numerical values
education_map = {"Undergrad": 1, "Graduate": 2, "Professional": 3}
cd_account_map = {"Yes": 1, "No": 0}
securities_account_map = {"Yes": 1, "No": 0}
online_banking_map = {"Yes": 1, "No": 0}
credit_card_map = {"Yes": 1, "No": 0}

# Applying the mappings to transform user inputs
education_level = education_map[education_level]
cd_account = cd_account_map[cd_account]
securities_account = securities_account_map[securities_account]
online_banking = online_banking_map[online_banking]
credit_card = credit_card_map[credit_card]

# Adjusting the scale for income, avg_credit_use, and mortgage to original data format
income = income / 1000
avg_credit_use = avg_credit_use / 1000
mortgage = mortgage / 1000

# Calculating the credit usage ratio
credit_usage_ratio = avg_credit_use / income if income > 0 else 0

# One-hot encoding for region input
region_columns = [
    "Region_Central",
    "Region_Los Angeles",
    "Region_Southern",
    "Region_Superior",
]
region_input = [1 if region == col.split(
    "_")[1] else 0 for col in region_columns]

# Combining all inputs into a single array
input_data = np.array(
    [
        experience,
        income,
        family_size,
        avg_credit_use,
        education_level,
        mortgage,
        securities_account,
        cd_account,
        online_banking,
        credit_card,
        credit_usage_ratio,
    ]
    + region_input
)

# Scaling the input data using the pre-loaded StandardScaler
scaled_input_data = StandardScaler.transform(input_data.reshape(1, -1))

############################################################################################

# UI for model selection and prediction
# Model selection
model_options = {
    "ROC AUC Optimized Model 📊": roc_auc_model,
    "F1 Weighted Optimized Model ⚖️": f1weight_model,
    "Recall Optimized Model 🔍": recall_model,
    "Precision Optimized Model 🎯": precision_model
}

selected_model_name = st.radio(
    "**Select a Model**", options=list(model_options.keys()), horizontal=True
)

# Predict
if st.button("Predict"):
    selected_model = model_options[selected_model_name]
    prediction = selected_model.predict(scaled_input_data)
    if prediction[0] == 1:
        st.success("The customer is likely to accept a personal loan.")
    else:
        st.error("The customer is unlikely to accept a personal loan.")

############################################################################################

# Data for displaying performance metrics of each model
roc_model_score = {
    "Metric": ["F1 Score", "Recall", "Precision"],
    "Training": [1.0, 1.0, 1.0],
    "Test": [0.946, 0.923, 0.970],
}

f1_model_score = {
    "Metric": ["F1 Score", "Recall", "Precision"],
    "Training": [0.975, 0.955, 0.996],
    "Test": [0.930, 0.923, 0.936],
}

recall_model_score = {
    "Metric": ["F1 Score", "Recall", "Precision"],
    "Training": [0.972, 0.955, 0.9907],
    "Test": [0.923, 0.9236, 0.923],
}

precision_model_score = {
    "Metric": ["F1 Score", "Recall", "Precision"],
    "Training": [0.281, 0.163, 1.0],
    "Test": [0.305, 0.180, 1.0],
}

# Convert dictionaries to DataFrames
df1 = pd.DataFrame(roc_model_score)
df2 = pd.DataFrame(f1_model_score)
df3 = pd.DataFrame(recall_model_score)
df4 = pd.DataFrame(precision_model_score)

# Function for displaying performance metrics in a table and images


def display_performance_metrics(title, conf_img, curve_img, df):
    st.header(title)
    st.table(df)
    st.image(f"images/{curve_img}")
    st.image(f"images/{conf_img}")

############################################################################################


# Define tabs for different models
tab1, tab2, tab3, tab4 = st.tabs([
    "ROC AUC Optimized Model 📊",
    "F1 Weighted Optimized Model ⚖️",
    "Recall Optimized Model 🔍",
    "Precision Optimized Model 🎯"
])


# Content for each tab: description and performance metrics
with tab1:
    st.write(
        """
        This model is a superstar 🌟 in distinguishing between customers who will 🙋‍♂️ and will not 🙅‍♂️ accept a personal loan. A high ROC AUC score means the model is a champ 🏆 at predicting true positives (customers saying "Yes, please!" to loans) without wrongly pointing at those who'll say "Nope!" 🚫. Opt for this model if you want top-notch accuracy in spotting potential loan accepters.
        """
    )
    if st.checkbox("Show ROC AUC Model Performance"):
        display_performance_metrics(
            "ROC AUC Optimized Model Performance 📊",
            "roc_conf.png",
            "roc_curve.png",
            df1,
        )
        st.markdown(
            """
            * **True Positives (TP)**: 133 sharp picks (8.87%) ✅ - Ace at spotting real loan takers!
            * **False Positives (FP)**: Just 4 little blips (0.27%) ❌ - Super precise with few missteps!
            * **True Negatives (TN)**: 1352 on-point no's (90.13%) 🎯 - Smart at identifying the not-interested.
            * **False Negatives (FN)**: 11 missed "yes" folks (0.73%) 🕵️ - Keen to catch every possible yes!
            """
        )
        st.write("### Model Parameters:")
        st.code(
            """
            make_pipeline(MaxAbsScaler(), GradientBoostingClassifier(max_depth=8, max_features=0.8,min_samples_leaf=13, min_samples_split=11, n_estimators=100, subsample=0.55, random_state=42))
            """
        )

# Tab content for F1 Weighted Optimized Model
with tab2:
    st.write(
        """
         The F1 Score is like a tightrope walker 🤹‍♂️, balancing precision (picking the right ones 🎯) and recall (not missing out on anyone 🕵️‍♂️). This model is your go-to if you're aiming for a sweet spot between finding as many "Yes" candidates as possible 🧲 while making sure most guesses are spot-on ✅. It's your buddy when both missing out and false alarms cost you dearly 💸.
        """
    )
    if st.checkbox("Show F1 Score Model Performance"):
        display_performance_metrics(
            "F1 Score Optimized Model Performance ⚖️",
            "f1_conf.png",
            "f1_curve.png",
            df2,
        )
        st.markdown(
            """
            * **True Positives (TP)**: 133 customers (8.87%) 👍 - Great at spotting loan yes-sayers!
            * **False Positives (FP)**: Only 9 mistaken calls (0.60%) ❌ - Keeps marketing costs in check!
            * **True Negatives (TN)**: 1347 right guesses (89.80%) 👏 - No wasting effort on the not-interested!
            * **False Negatives (FN)**: Missed 11 potential yes-sayers (0.73%) 🕵️‍♂️ - Aiming to catch more!
            """
        )
        st.write("### Model Parameters:")
        st.code(
            """
            GradientBoostingClassifier(learning_rate=0.5, max_depth=4, max_features=0.15,min_samples_leaf=7, min_samples_split=10, n_estimators=100, subsample=0.9, random_state=42)
            """
        )

# Tab content for Recall Optimized Model
with tab3:
    st.write(
        """
         The Recall-optimized model is like a wide net 🌐, aiming to catch nearly everyone who would nod yes to a loan. If you're more worried about missing a "Yes" than catching a "Maybe," this is your pick 🎣. It's perfect if missing a potential "Yes" is a big no-no 🚫.
        """
    )
    if st.checkbox("Show Recall Model Performance"):
        display_performance_metrics(
            "Recall Optimized Model Performance 🔍",
            "recall_conf.png",
            "recall_curve.png",
            df3)
        st.markdown(
            """
            * **True Positives (TP)**: 141 right picks (9.40%) ✅ - Keen on catching interested folks!
            * **False Positives (FP)**: 468 oopsies (31.20%) 🙈 - A bit eager, but that's the strategy!
            * **True Negatives (TN)**: 888 correct no’s (59.20%) 🚫 - Still avoids bothering the uninterested.
            * **False Negatives (FN)**: Only 3 missed chances (0.20%) 👀 - Nearly misses no one
            """
        )
        st.write("### Model Parameters:")
        st.code(
            """
            SGDClassifier(alpha=0.0, eta0=0.01, fit_intercept=False, l1_ratio=0.25 learning_rate="constant", loss="log_loss", penalty="elasticnet", power_t=100.0, random_state=42)
            """
        )

# Tab content for Precision Optimized Model
with tab4:
    st.write(
        """
         Tuned for Precision, this model is like a sniper 🔫— highly accurate in pinpointing customers ready to say "I'm in!" for a loan. It's a bit cautious and might skip some potential yes-sayers to avoid the "No, thanks" crowd. Pick this if you're all for precision and want to minimize the "Oops, wrong person" moments 🙊, even if it means missing out on a few hidden gems 💎.
        """
    )
    if st.checkbox("Show Precision Model Performance"):
        display_performance_metrics(
            "Precision Optimized Model Performance 🎯", "precision_conf.png",
            "precision_curve.png",
            df4,
        )
        st.markdown(
            """
            * **True Positives (TP)**: 26 accurate hits (1.73%) 🎖️ - Choosy but precise!
            * **False Positives (FP)**: Zero wrong calls (0.00%) 🎉 - Spot-on every time!
            * **True Negatives (TN)**: 1356 correct non-targets (90.40%) 🛡️ - Avoids the uninterested well.
            * **False Negatives (FN)**: 118 missed loan-takers (7.87%) 🧐 - Trades off some misses for accuracy
            """
        )
        st.write("### Model Parameters:")
        st.code("KNeighborsClassifier(n_neighbors=71, p=1)")
