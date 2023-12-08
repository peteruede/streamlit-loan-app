# Libraries to help with reading and manipulating data
import pandas as pd
import numpy as np
import zipcodes as zcode
import seaborn as sns
import altair as alt
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px

############################################################################################

st.set_page_config(
    page_title="Data Explorer",
    page_icon="📊",
    menu_items={
        "About": """This web application is a project that aims to assist AllLife Bank in identifying potential customers 
        likely to accept personal loan offers. It is based on a dataset of customer demographics and financial behavior."""
    },
)

# Streamlit page configuration
sns.set_style("white")
st.set_option("deprecation.showPyplotGlobalUse", False)

############################################################################################

# Function to load and cache the data
file = "Loan_Modelling.csv"


@st.cache_data
def load_data(file):
    cached_data = pd.read_csv(file)
    return cached_data


cached_data = load_data(file)
df = cached_data.copy()

############################################################################################

# Application title and introduction
st.title("🏦 Bank Loan Marketing Strategy")
st.image("images/dollar.webp")
st.markdown(
    """    
**AllLife Bank is on a mission to 📈 expand its customer base. The goal is to target potential clients who are likely candidates for personal loans.**

📱 **The App's Role**:
    
* Data-Driven Marketing : Utilize data from a previous kickstart campaign to gain insights 📊 into customer behavior.
* Customer Profiling: Input customer profiles to generate predictions 🔮 on their likelihood 📈📉 of accepting personal loan offers.
    
🔍 **Dataset Overview**:

The dataset used 📁 contains details on 5,000 customers from AllLife Bank, focusing on their responses to personal loan offers in a past campaign.
"""
)

with st.expander("📚 **Data Points**"):
    st.write(
        """
        * ID 🆔: Unique identification number for each customer.
        * Age 🎂: Customer's age in years.
        * Experience 💼: Years of professional experience (corrected for negatives).
        * Income 💲: Annual income in thousands of dollars.
        * ZIP Code 📍: Customer's home address ZIP code.
        * Family 👨‍👩‍👧‍👦: Family size (up to 4 members).
        * Avg Credit Card Use 💳: Average monthly credit card spending in thousands.
        * Education 🎓: Level of education (1-Undergraduate, 2-Graduate, 3-Professional).
        * Mortgage 🏠: House mortgage value in thousands.
        * Securities Account 📈: Presence of a securities account with the bank.
        * CD Account 💿: Certificate of deposit with the bank.
        * Online 🌐: Use of internet banking facilities.
        * Credit Card 💳: Use of a credit card issued by AllLife Bank.
        * Personal Loan 🏦: Acceptance of a personal loan in the last campaign.
         """
    )

############################################################################################

# Data Preprocessing
# --------------------------------

# Clean columns names
rename_columns = {
    "CCAvg": "Avg Credit Card Use",
    "CD_Account": "CD Account",
    "CreditCard": "Credit Card",
    "Personal_Loan": "Personal Loan",
    "Securities_Account": "Securities Account",
}
df.rename(columns=rename_columns, inplace=True)

# Replacing negative 'Experience' values with global median
median_exp = df["Experience"].median()
df["Experience"].fillna(median_exp, inplace=True)

# Converting Zipcode to County
list_zipcode = df.ZIPCode.unique()
dict_zip = {}
for zipcode in list_zipcode:
    city_county = zcode.matching(zipcode.astype("str"))
    if len(city_county) == 1:
        county = city_county[0].get("county")
    else:
        county = zipcode

    dict_zip.update({zipcode: county})
    dict_zip.update({92717: "Orange County"})
    dict_zip.update({92634: "Orange County"})
    dict_zip.update({96651: "El Dorado County"})
    dict_zip.update({93077: "Ventura County"})


# Converting the county to regions based on https://www.calbhbc.org/region-map-and-listing.html
counties = {
    "Los Angeles County": "Los Angeles",
    "San Diego County": "Southern",
    "Santa Clara County": "Bay Area",
    "Alameda County": "Bay Area",
    "Orange County": "Southern",
    "San Francisco County": "Bay Area",
    "San Mateo County": "Bay Area",
    "Sacramento County": "Central",
    "Santa Barbara County": "Southern",
    "Yolo County": "Central",
    "Monterey County": "Bay Area",
    "Ventura County": "Southern",
    "San Bernardino County": "Southern",
    "Contra Costa County": "Bay Area",
    "Santa Cruz County": "Bay Area",
    "Riverside County": "Southern",
    "Kern County": "Southern",
    "Marin County": "Bay Area",
    "San Luis Obispo County": "Southern",
    "Solano County": "Bay Area",
    "Humboldt County": "Superior",
    "Sonoma County": "Bay Area",
    "Fresno County": "Central",
    "Placer County": "Central",
    "Butte County": "Superior",
    "Shasta County": "Superior",
    "El Dorado County": "Central",
    "Stanislaus County": "Central",
    "San Benito County": "Bay Area",
    "San Joaquin County": "Central",
    "Mendocino County": "Superior",
    "Tuolumne County": "Central",
    "Siskiyou County": "Superior",
    "Trinity County": "Superior",
    "Merced County": "Central",
    "Lake County": "Superior",
    "Napa County": "Bay Area",
    "Imperial County": "Southern",
}

############################################################################################
# Feature Engineering

# Add County to the dataset then drop Zipcode
df["County"] = df["ZIPCode"].map(dict_zip)
df.drop("ZIPCode", axis=1, inplace=True)

df["Region"] = df["County"].map(counties)

# Create AgeGroup by binning age
df["Age Group"] = pd.cut(
    df["Age"],
    bins=[0, 25, 40, 55, 70, np.inf],
    labels=["<25", "26-40", "41-55", "56-70", "70+"],
)

# Create Income Class by binning Income
df["Income Group"] = pd.cut(
    df["Income"],
    bins=[0, 30, 60, 90, 120, np.inf],
    labels=["Low", "Below Average", "Average", "Above Average", "High"],
)

# Create Credit Usage Ratio
df["Credit Usage Ratio"] = df["Avg Credit Card Use"] / df["Income"]

# Define mappings for the conversions
conversion_mappings = {
    "Securities Account": {1: "Yes", 0: "No"},
    "CD Account": {1: "Yes", 0: "No"},
    "Online": {1: "Yes", 0: "No"},
    "Credit Card": {1: "Yes", 0: "No"},
    "Personal Loan": {1: "Yes", 0: "No"},
    "Education": {1: "Undergrad", 2: "Graduate", 3: "Professional"},
    "Family": {1: "Single", 2: "Small", 3: "Medium", 4: "Large"},
}

# Apply the mappings and convert selected columns to categorical variables
for column, mapping in conversion_mappings.items():
    df[column] = df[column].map(mapping).astype("category")

check_data = st.toggle("Show the Original Dataset")
if check_data:
    start, end = st.slider(
        "Select number of rows to display 🔢", 1, len(cached_data), (1, 5), key="sld1"
    )
    st.dataframe(cached_data.iloc[start - 1: end], width=800)

# Features Creation
st.subheader("🛠️ Feature Creation", divider="blue")
geo_cont = st.container()
with geo_cont:
    # Create layout columns
    col1, col2 = st.columns(2)

    # Display the Counties created
    with col1:
        with st.expander("##### Converted Zip Codes 📍 to County 🏘️"):
            st.dataframe(df["County"].value_counts(), width=300)

    # Display the Regions created
    with col2:
        with st.expander("##### Created Region 🌍 from Counties 🏞️"):
            st.dataframe(df["Region"].value_counts(), width=300)

agegroup_cont = st.container()
with agegroup_cont:
    # Create more layout columns
    col3, col4 = st.columns(2)

    # Display the Age Groups created
    with col3:
        with st.expander("##### Created Age Group 👶🧓 from Age 🎂"):
            st.dataframe(df["Age Group"].value_counts(), width=300)

    # Display the Income Groups created
    with col4:
        with st.expander("##### Created Income Group 💼 from Income 💵"):
            st.dataframe(df["Income Group"].value_counts(), width=300)


# Button that allows the user to see the entire table
check_df = st.toggle("Show the New Dataset")
if check_df:
    start, end = st.slider(
        "Select number of rows to display 🔢", 1, len(df), (1, 5), key="sld2")
    st.dataframe(df.iloc[start-1:end], width=800)


# Use Streamlit's metric function to display key figures
st.metric(
    label="Total Personal Loans Accepted", value=int(sum(df["Personal Loan"] == "Yes"))
)

############################################################################################

# Separate the columns into numeric and non-numeric excluding 'County'
numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
non_numeric_columns = df.select_dtypes(exclude=np.number).columns.tolist()
non_numeric_columns.remove("County")

# Interactive Distribution Plot
st.subheader(
    "🔍Explore Customer Demographics & Banking Attributes", divider="blue")

# Define available variables for X, Color, and Facet
x_variables = ["Age", "Income", "Avg Credit Card Use", "Experience"]
color_variables = ["Personal Loan", "Securities Account", "CD Account"]
facet_variables = non_numeric_columns + ["None"]

# Define the desired order for Family and Education based on selection
family_order = ["Single", "Small", "Medium", "Large"]
education_order = ["Undergrad", "Graduate", "Professional"]

col5, col6, col7 = st.columns(3)
with col5:
    # Selectbox for X variable
    x_variable = st.selectbox(
        "**Select Customer Attribute:**", x_variables, index=x_variables.index("Income")
    )

with col6:
    # Selectbox for Color variable with a default "None" option
    color_variable = st.selectbox("**Bank Services:**", color_variables)

with col7:
    # Selectbox for Facet variable with a default "None" option
    facet_variable = st.selectbox(
        "**Choose Variable for Subplot:**",
        facet_variables,
        index=non_numeric_columns.index("Family") + 1,
    )

click_selection = alt.selection_point(encodings=["color"])

# Define the base chart with common properties
base_dist_chart = (
    alt.Chart(df)
    .mark_bar()
    .encode(
        x=alt.X(f"{x_variable}:Q", bin=alt.Bin(maxbins=25)),
        y=alt.Y("count()", title="Count"),
        color=alt.Color(
            f"{color_variable}:N", scale=alt.Scale(range=["lightgray", "blue"])
        ),
        tooltip=[x_variable, color_variable]
        + ([facet_variable] if facet_variable != "None" else []),
        opacity=alt.condition(click_selection, alt.value(0.9), alt.value(0.1)),
    )
    .add_params(click_selection)
)

# Define the faceted chart
if facet_variable != "None":
    if facet_variable == "Family":
        sort_order = family_order
    elif facet_variable == "Education":
        sort_order = ["Undergrad", "Graduate", "Professional"]
    else:
        sort_order = None  # For other categorical variables, no specific sort order

    # Facet the chart with the specified number of columns
    dist_chart = base_dist_chart.facet(
        facet=alt.Facet(f"{facet_variable}:N", sort=sort_order), columns=2
    ).resolve_scale(y="independent")
else:
    dist_chart = base_dist_chart.properties(width=800, height=500)

st.altair_chart(dist_chart)

# ############################################################################################

# Interactive Scatterplot
st.subheader("💼 Income and Credit Use Analysis", divider="blue")
st.write(
    """Income is often a significant indicator in financial behavior, including the propensity 
    to take out loans. The scatterplot below allows for a detailed examination of the relationship 
    between customers' income and their use of credit facilities."""
)
st.write("Explore the relationship between people's income and credit use 💳")

col8, col9 = st.columns(2)
with col8:
    x_dropdown = st.selectbox(
        "**Choose Credit Type:**", ["Avg Credit Card Use", "Mortgage"]
    )

with col9:
    color_dropdown = st.selectbox(
        "**Choose category to compare:**",
        ["None", "Personal Loan", "Education", "CD Account"],
        index=1,
    )

scatter_chart = (
    (
        alt.Chart(df)
        .mark_point(size=60)
        .encode(
            alt.X(f"{x_dropdown}"),
            alt.Y("Income"),
            color=alt.Color(
                f"{color_dropdown}:N",
                scale=alt.Scale(scheme="set1", reverse=True),
                legend=alt.Legend(title=color_dropdown),
            )
            if color_dropdown != "None"
            else alt.value("gray"),
            opacity=alt.condition(
                click_selection, alt.value(0.9), alt.value(0.01)),
        )
        .add_params(click_selection)
    )
    .properties(width=750, height=550)
    .interactive()
)

st.altair_chart(scatter_chart)

# Display correlation metric
st.write("### Correlation Analysis 🔗📈")
correlation = round(df[x_dropdown].corr(df["Income"]), 2)
st.metric(f"Correlation between {x_dropdown} and Income:", correlation)

# Interpret the correlation value
if correlation <= -0.5:
    st.write("There is a strong negative correlation.")
elif correlation <= -0.1:
    st.write("There is a moderate negative correlation.")
elif correlation < 0.1:
    st.write("There is no significant correlation.")
elif correlation < 0.5:
    st.write("There is a moderate positive correlation.")
elif correlation >= 0.5:
    st.write("There is a strong positive correlation.")

# Interesting Findings
############################################################################################

# Key findings related to mortgages
st.header("🔑 Key Findings", divider="blue")
st.subheader("Insights on Mortgages and Loans 🏠💰")

# Explanation about this section
st.write(
    """
         The density plot below depicts how mortgage values distribute among those with and without personal loans, 
         highlighting the patterns and tendencies in loan acceptance relative to mortgage amounts. 
         **People with lower mortgages tend not to take out more loans**"""
)

# Interval selection for interaction with the density plot
selection = alt.selection_point(encodings=["y"])

# Create a density chart for mortgage distribution
mortgage_density_chart = (
    alt.Chart(df[df["Mortgage"] > 0], title=alt.Title(
        "Mortgage Distribution Analysis"))
    .transform_density(
        "Mortgage", as_=["Mortgage", "density"], groupby=["Personal Loan"]
    )
    .mark_area(fillOpacity=0.7)
    .encode(
        alt.X(
            "Mortgage:Q", title="Mortgage (in thousands)", axis=alt.Axis(format="$.0f")
        ),
        alt.Y("density:Q").axis(None),
        color=alt.Color(
            "Personal Loan:O",
            legend=alt.Legend(title="Has Personal Loan", symbolType="square"),
            scale=alt.Scale(domain=["Yes", "No"], range=["blue", "lightgrey"]),
        ),
        opacity=alt.condition(selection, alt.value(0.8), alt.value(0.01)),
        tooltip=[alt.Tooltip("Personal Loan"), "Mortgage"],
    )
    .properties(height=500, width=700)
    .configure_axis(grid=False)
    .add_params(selection)
)

st.altair_chart(mortgage_density_chart)
############################################################################################

# Plot Income distribution
st.subheader("Income and Credit Card Usage Analysis 💳")

# Create a click selection
click = alt.selection_point(encodings=["color"])

# Create the income distribution chart
income_density_chart = (
    alt.Chart(
        df,
        title=alt.Title(
            "Income Distribution Analysis",
            subtitle=[
                "People earning above $60,000 are more likely to take out loans.",
                "Majority of loan recipients earn around $120,000 or $180,000.",
            ],
        ),
    )
    .transform_density("Income", as_=["Income", "density"], groupby=["Personal Loan"])
    .mark_area(fillOpacity=0.7)
    .encode(
        x=alt.X(
            "Income:Q",
            title="Customer Income (in thousands)",
            axis=alt.Axis(format="$.0f", ticks=False),
        ),
        y=alt.Y("density:Q", axis=None),
        color=alt.Color(
            "Personal Loan:O",
            scale=alt.Scale(domain=["Yes", "No"], range=["blue", "lightgrey"]),
            legend=None,
        ),
        opacity=alt.condition(click, alt.value(0.8), alt.value(0.01)),
        tooltip=[
            alt.Tooltip("Personal Loan:O", title="Personal Loan"),
            alt.Tooltip("Income:Q", title="Income (in thousands)"),
        ],
    )
    .add_params(click)
    .properties(height=400, width=400)
)


# Create the CCAvg distribution chart
density_with_ccavg = (
    alt.Chart(
        df,
        title=alt.Title(
            "Monthly Credit Card Usage by Personal Loan Status",
            subtitle=[
                " Most loan recipients spend around $3,000 to $6,000 on credit cards."
            ],
        ),
    )
    .transform_density(
        "Avg Credit Card Use",
        as_=["Avg Credit Card Use", "density"],
        groupby=["Personal Loan"],
    )
    .mark_area(fillOpacity=0.7)
    .encode(
        x=alt.X(
            "Avg Credit Card Use:Q",
            title="Average Credit Card Spending per Month (in thousands)",
            axis=alt.Axis(format="$.0f", ticks=False),
        ),
        y=alt.Y("density:Q").axis(None),
        color=alt.Color(
            "Personal Loan:O",
            scale=alt.Scale(domain=["Yes", "No"], range=["blue", "lightgrey"]),
            legend=alt.Legend(orient="top-right", title=None,
                              symbolType="square"),
        ),
        opacity=alt.condition(click, alt.value(0.8), alt.value(0.01)),
        tooltip=[
            alt.Tooltip("Personal Loan:O", title="Personal Loan"),
            alt.Tooltip(
                "Avg Credit Card Use:Q", title="Credit Card Use (in thousands)"
            ),
        ],
    )
    .add_params(click)
    .properties(height=400, width=400)
)


# Create the boxplot chart (boxy) for Income
boxy_income = (
    alt.Chart(
        df,
        title=alt.Title(
            "Boxplot of Income by Personal Loan Status",
            subtitle="Boxplots provide a summary of the distributions and can point out outliers.",
        ),
    )
    .mark_boxplot()
    .encode(
        x=alt.X(
            "Income:Q",
            title="Customer Income (in thousands)",
            axis=alt.Axis(format="$.0f"),
        ),
        y=alt.Y("Personal Loan:O", axis=alt.Axis(orient="right")),
        color=alt.Color("Personal Loan:N", title="Personal Loan"),
        opacity=alt.condition(click, alt.value(0.8), alt.value(0.01)),
    )
    .properties(height=150, width=400)
)

# Create the boxplot chart (boxy) for Avg Credit Card Use
boxy_ccavg = (
    alt.Chart(
        df, title=alt.Title(
            "Boxplot of Credit Card Spending by Personal Loan Status")
    )
    .mark_boxplot()
    .encode(
        y=alt.Y("Personal Loan:O", axis=alt.Axis(orient="right")),
        x=alt.X(
            "Avg Credit Card Use:Q",
            title="Average Credit Card Spending per Month (in thousands)",
            axis=alt.Axis(format="$.0f", ticks=False),
        ),
        color=alt.Color("Personal Loan:N", title="Personal Loan"),
        opacity=alt.condition(click, alt.value(0.8), alt.value(0.01)),
    )
    .properties(height=150, width=400)
)

# Concatenate the two boxplots horizontally
boxy = alt.vconcat(
    boxy_income,
    boxy_ccavg,
)


# Create the scatterplot of Income vs. CCAvg with click interaction
scatterplot = (
    alt.Chart(
        df,
        title=alt.Title(
            "Income vs. Credit Card Spending by Personal Loan Status",
            subtitle=[
                "This scatterplot helps us understand if there's a direct relationship between income and",
                "credit card expenditure for those with personal loans. Loan recipients typically have",
                "high annual income. They also spend a lot on credit cards.",
            ],
        ),
    )
    .mark_point()
    .encode(
        y=alt.X(
            "Income:Q",
            title="Customer Income (in thousands)",
            axis=alt.Axis(format="$.0f", orient="left"),
        ),
        x=alt.Y(
            "Avg Credit Card Use:Q",
            title="Average Credit Card Spending per Month (in thousands)",
            axis=alt.Axis(format="$.0f"),
        ),
        color=alt.Color("Personal Loan:N", title="Personal Loan"),
        opacity=alt.condition(click, alt.value(0.8), alt.value(0.01)),
    )
    .add_params(click)
    .properties(height=400, width=400)
    .interactive()
)

# Display the layered chart
st.altair_chart((income_density_chart | density_with_ccavg)
                & (scatterplot | boxy))

############################################################################################
custom_palette = {"Yes": "Blue", "No": "grey"}

# Stripplot for Credit Card Usage vs. CD Account
st.subheader("Certificate of Deposit Account Analysis 📜💰")
st.write(
    """
         It seems that customers with higher credit card spending and higher incomes are more inclined to possess a Certificate of Deposit (CD) Account. 
         Financial institutions might leverage this insight to target potential CD account holders 💡.
         """
)

plt.figure(figsize=(10, 7))
sns.stripplot(
    x="CD Account",
    y="Avg Credit Card Use",
    hue="Personal Loan",
    data=df,
    dodge=True,
    palette=custom_palette,
)

plt.title(
    "Avg Credit Card Usage of Customers with and without a Certificate of Deposit Account",
    fontsize=16,
)
plt.legend(title="Personal Loan", loc="upper right", frameon=False)
plt.xlabel("CD Account", fontsize=14)
plt.ylabel("Avg Credit Card Use", fontsize=14)
plt.tick_params(axis="both", labelsize=13)

st.pyplot(plt)


# Stripplot for Income vs. CD Account
st.write("High-income earners might be ideal candidates for Certificate of Deposit Accounts given their financial capability 💰🎯")

plt.figure(figsize=(10, 7))
sns.stripplot(
    x="CD Account",
    y="Income",
    hue="Personal Loan",
    data=df,
    dodge=True,
    palette=custom_palette,
)

plt.title(
    "Annual Income of Customers with and without a Certificate of Deposit Account",
    fontsize=16,
)
plt.legend(title="Personal Loan", loc="upper right", frameon=False)
plt.xlabel("CD Account", fontsize=14)
plt.ylabel("Income", fontsize=14)
plt.tick_params(axis="both", labelsize=13)

st.pyplot(plt)


# Stripplot for Family Size vs. Income
st.subheader("👨‍👩‍👧‍👦 Family Size Analysis")
st.write(
    "Family size appears to play a role in loan acceptance, particularly for those earning above $50,000 annually."
)

plt.figure(figsize=(10, 7))
sns.stripplot(
    x="Family",
    y="Income",
    hue="Personal Loan",
    data=df,
    dodge=True,
    palette=custom_palette,
    order=family_order,
)

plt.title("Family Size and Annual Income Relationship", fontsize=16)
plt.legend(title="Personal Loan", loc="upper right", frameon=False)
plt.xlabel("Family", fontsize=14)
plt.ylabel("Income", fontsize=14)
plt.tick_params(axis="both", labelsize=13)

st.pyplot(plt)

tab1, tab2 = st.tabs(["🕵️Insights", "🌟Recommendations"])
with tab1:
    st.markdown(
        """
        * 🔑 **Influential Variables**: **Income, family size, education, credit card spend, CD account** - These are the MVPs 🏆 in predicting who's up for a personal loan. They're like the secret sauce 🍲 to understanding financial health and loan-taking vibes.

        * 🎓 **Impact of Education**: More education often means more "Yes" to loans! 🙋‍♀️ Why? Maybe because of fatter paychecks 💸, smarty-pants financial skills 🧠, and a VIP pass 🎟️ to banking products.

        * 💳 **Credit Card Spending**: Love swiping your credit card? You might also be game for personal loans! This trend hints that if you're cool with credit, you're likely to welcome more financial adventures. 

        * 💵 **Income Correlation with Loan Feasibility**: Making it rain with a high income? Then, personal loans might be on your radar. It's all about having enough dough 🍞 to comfortably pay back.

        * 🧐 **Less Predictive Variables**: Age, experience, zip code - Not the big cheeses 🧀 when it comes to loan-taking. It seems your money moves 💃 matter more than your years or your map pin. 📍

        * 👨‍🎓 **Graduate Propensity for Loans**: Graduates are more likely to hop onto the loan train 🚂. Maybe it's because they've got better financial tools 🔧 or more $$$ needs due to school bills. 🏫

        * 🚦 **Income Threshold for Loan Rejection**: Earning less than $80K? Loans might be a no-go zone. 🚫 This could be playing it safe 🛡️ from extra financial burdens for those on a tighter budget. 💼
        """
    )

with tab2:
    st.markdown(
        """
    **👨‍👩‍👧 Targeting Based on Family Size and Income:**

    * Eye on small families (2 or less) with big incomes 💰. Despite their heavy credit card use, they're not big on loans yet. 🤔 Time for some tailored marketing magic ✨ to show them how a personal loan fits into their chic lifestyle. 🕶️

    **🛍️ Targeting Middle Class Spenders:**

    * Those in the middle-income range but swiping their cards like there's no tomorrow 🌪️ are perfect for personal loans. Let's roll out the red carpet with offers that scream ease and benefits.

    **💳 High-Spenders on Credit Cards:**

    * Folks dropping $3,800+ on credit cards monthly? They're ripe for personal loans! 🍇 Strategy: Woo them with ideas that resonate with their spend-smart mindset. 💡

    **👨‍👩‍👧‍👦 Focusing on Large Families with High Income:**

    * Big families with incomes over $100K 💵 are like hidden treasure chests for loans. 🏴‍☠️ Campaigns should sing 🎶 about managing family finances and gearing up for big expenses with personal loans. 🎁

    **🏦 CD Account Holders as Prime Targets:**

    * CD account owners? They're the gold standard for loan takers. 🏆 Already financially savvy 🧠, they'll likely bite if you serve up competitive loan deals. Let's get them on board! 🚀
    """
    )
