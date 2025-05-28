import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.express as px

# Đọc và xử lý dữ liệu
def load_data():
    df = pd.read_csv("Mall_Customers.csv")
    df.columns = [col.strip().replace(" ", "_") for col in df.columns]
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    elif 'Genre' in df.columns:
        df['Gender'] = df['Genre'].map({'Male': 0, 'Female': 1})
    else:
        raise KeyError("Không có cột Gender hoặc Genre.")
    return df

# Dự đoán điểm chi tiêu (regression)
def predict_spending(df):
    df = load_data()
    X = df[['Gender', 'Age', 'Annual_Income_(k$)']]
    y = df['Spending_Score_(1-100)']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    df['Predicted_Spending'] = model.predict(X).round(2)
    return df

# Phân cụm khách hàng
def cluster_customers(df):
    df = load_data()
    features = ['Age', 'Annual_Income_(k$)', 'Spending_Score_(1-100)']
    X_scaled = StandardScaler().fit_transform(df[features])
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    fig3d = px.scatter_3d(df, x='Age', y='Annual_Income_(k$)', z='Spending_Score_(1-100)', color='Cluster', symbol='Gender', title="Phân nhóm khách hàng (3D)")
    fig2d = px.scatter(df, x='Annual_Income_(k$)', y='Spending_Score_(1-100)', color='Cluster', title="Scatter: Thu nhập vs Điểm tiêu dùng")
    return df, fig3d, fig2d

# Phân loại khách hàng: high-spending (>=50) vs low (<50)
def classify_customers(df):
    df = load_data()
    df['Label'] = (df['Spending_Score_(1-100)'] >= 50).astype(int)
    X = df[['Gender', 'Age', 'Annual_Income_(k$)']]
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    df['Prediction_Class'] = clf.predict(X)
    return df

# Biểu đồ Histogram
def create_histograms(df):
    df = load_data()
    histograms = []
    for col in ['Age', 'Annual_Income_(k$)', 'Spending_Score_(1-100)']:
        fig = px.histogram(df, x=col, nbins=20, title=f"Histogram: {col}")
        histograms.append(fig)
    return histograms