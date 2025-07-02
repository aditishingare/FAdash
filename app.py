
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from mlxtend.frequent_patterns import apriori, association_rules

@st.cache_data
def load_data(url=None, file=None):
    if file is not None:
        return pd.read_csv(file)
    elif url:
        return pd.read_csv(url)
    else:
        st.error("Provide GitHub raw CSV URL or upload file.")
        return None

st.set_page_config(page_title="Fashion Rental Dashboard", layout="wide")
st.title("Sustainable Fashion-Rental App Dashboard")

# Input for data source
st.sidebar.header("Data Source")
data_url = st.sidebar.text_input("GitHub Raw CSV URL", "")
uploaded_file = st.sidebar.file_uploader("Or upload CSV", type="csv")

df = load_data(data_url, uploaded_file)
if df is None:
    st.stop()

# Tabs
tabs = st.tabs(["Data Visualization", "Classification", "Clustering", "Association Rules", "Regression"])

# ---------------- Data Visualization ----------------
with tabs[0]:
    st.header("Data Visualization Insights")
    # 1. Age distribution
    fig, ax = plt.subplots()
    df['Q1_Age'].hist(bins=20, ax=ax)
    ax.set_title("Age Distribution")
    st.pyplot(fig)
    # 2. Income distribution
    fig, ax = plt.subplots()
    df['Q3_IncomeUSD'].hist(bins=20, ax=ax)
    ax.set_title("Income Distribution")
    st.pyplot(fig)
    # 3. Gender count
    st.bar_chart(df['Q2_Gender'].value_counts())
    # 4. Monthly spend vs income scatter
    fig, ax = plt.subplots()
    ax.scatter(df['Q3_IncomeUSD'], df['Q19_MonthlyRentalSpendUSD'], alpha=0.5)
    ax.set_title("Spend vs Income")
    ax.set_xlabel("Income USD")
    ax.set_ylabel("Monthly Rental Spend USD")
    st.pyplot(fig)
    # 5. Items per month boxplot
    fig, ax = plt.subplots()
    df.boxplot(column='Q20_ItemsPerMonth', ax=ax)
    ax.set_title("Items Per Month Distribution")
    st.pyplot(fig)
    # 6. Trigger frequency pie
    st.write("Trigger Frequency Breakdown")
    freq = df['Q7_TriggerFrequency'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(freq, labels=freq.index, autopct='%1.1f%%')
    st.pyplot(fig)
    # 7. Pricing model count
    st.bar_chart(df['Q18_PricingModel'].value_counts())
    # 8. Subscription likelihood histogram
    fig, ax = plt.subplots()
    df['Q25_SubscribeLikelihood'].hist(bins=10, ax=ax)
    ax.set_title("Subscription Likelihood")
    st.pyplot(fig)
    # 9. NPS distribution
    fig, ax = plt.subplots()
    df['Q29_RecommendNPS'].hist(bins=10, ax=ax)
    ax.set_title("NPS Distribution")
    st.pyplot(fig)
    # 10. Correlation heatmap
    num_cols = ['Q1_Age','Q3_IncomeUSD','Q19_MonthlyRentalSpendUSD','Q20_ItemsPerMonth','Q25_SubscribeLikelihood','Q29_RecommendNPS']
    corr = df[num_cols].corr()
    fig, ax = plt.subplots()
    import seaborn as sns
    sns.heatmap(corr, annot=True, ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

# ---------------- Classification ----------------
with tabs[1]:
    st.header("Classification Algorithms")
    # Prepare data: Use Q25_SubscribeLikelihood>=5 as target for binary classification
    X = df[['Q1_Age','Q3_IncomeUSD','Q19_MonthlyRentalSpendUSD','Q20_ItemsPerMonth']]
    y = (df['Q25_SubscribeLikelihood'] >= 5).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # Models
    models = {
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'GBRT': GradientBoostingClassifier(random_state=42)
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-score': f1_score(y_test, y_pred)
        }
    res_df = pd.DataFrame(results).T
    st.dataframe(res_df.style.format("{:.2f}"))
    # Confusion matrix toggle
    algo = st.selectbox("Select Algorithm for Confusion Matrix", list(models.keys()))
    cm = confusion_matrix(y_test, models[algo].predict(X_test))
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {algo}")
    st.pyplot(fig)
    # ROC curves
    fig, ax = plt.subplots()
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
    ax.plot([0,1], [0,1], 'k--')
    ax.set_title("ROC Curves")
    ax.legend(loc='lower right')
    st.pyplot(fig)
    # Upload new data
    new_data = st.file_uploader("Upload new data for prediction (CSV)", type="csv", key="class")
    if new_data:
        new_df = pd.read_csv(new_data)
        preds = models[algo].predict(new_df[X.columns])
        out = new_df.copy()
        out['predicted_label'] = preds
        st.write(out.head())
        st.download_button("Download Predictions", out.to_csv(index=False), file_name="classification_results.csv")

# ---------------- Clustering ----------------
with tabs[2]:
    st.header("K-Means Clustering")
    # Elbow chart
    inertias = []
    K_range = range(1, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42).fit(X)
        inertias.append(km.inertia_)
    fig, ax = plt.subplots()
    ax.plot(K_range, inertias, marker='o')
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Chart")
    st.pyplot(fig)
    # Slider for clusters
    k = st.slider("Select number of clusters", 2, 10, 3)
    km = KMeans(n_clusters=k, random_state=42).fit(X)
    df['cluster'] = km.labels_
    # Persona table: cluster centroids
    persona = pd.DataFrame(km.cluster_centers_, columns=X.columns)
    persona['cluster'] = persona.index
    st.write("Cluster Centers (Persona Profiles)")
    st.dataframe(persona)
    # Download cluster-labeled data
    st.download_button("Download Clustered Data", df.to_csv(index=False), file_name="clustered_data.csv")

# ---------------- Association Rules ----------------
with tabs[3]:
    st.header("Association Rule Mining")
    cols = st.multiselect("Select columns for Apriori (multi-select)", ['Q6_Triggers','Q15_CompetingSolutions'])
    min_sup = st.slider("Min Support", 0.01, 0.5, 0.1)
    min_conf = st.slider("Min Confidence", 0.1, 1.0, 0.3)
    if cols:
        # One-hot encode
        df_onehot = pd.get_dummies(df[cols].apply(lambda x: x.str.split(', ')).explode(cols).reset_index(), 
                                   prefix=cols, prefix_sep='_', 
                                   columns=cols).groupby('index').max()
        freq = apriori(df_onehot, min_support=min_sup, use_colnames=True)
        rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
        top10 = rules.sort_values('confidence', ascending=False).head(10)
        st.dataframe(top10[['antecedents','consequents','support','confidence','lift']])

# ---------------- Regression ----------------
with tabs[4]:
    st.header("Regression Models")
    y_reg = df['Q19_MonthlyRentalSpendUSD']
    X_reg = df[['Q1_Age','Q3_IncomeUSD','Q20_ItemsPerMonth']]
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)
    regressors = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'Decision Tree': DecisionTreeRegressor(random_state=42)
    }
    reg_results = {}
    for name, reg in regressors.items():
        reg.fit(Xr_train, yr_train)
        pred = reg.predict(Xr_test)
        rmse = np.sqrt(((pred - yr_test) ** 2).mean())
        r2 = reg.score(Xr_test, yr_test)
        reg_results[name] = {'RMSE': rmse, 'R2': r2}
        # Plot actual vs predicted
        fig, ax = plt.subplots()
        ax.scatter(yr_test, pred, alpha=0.5)
        ax.set_xlabel("Actual Spend")
        ax.set_ylabel("Predicted Spend")
        ax.set_title(f"{name} Regression")
        st.pyplot(fig)
    st.write(pd.DataFrame(reg_results).T.style.format("{:.2f}"))
