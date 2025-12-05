
import pickle
import streamlit as st
import pandas as pd
import sklearn
from sklearn.metrics import r2_score, mean_squared_error as MSE, mean_absolute_error as MAE
import re
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import phik


class MyTransormer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        self.regex_pattern = r"[-+]?([0-9]*\.[0-9]+|\d+)" 

    def fit(self, X_real, y_real):
        X = X_real.copy(deep=True)
        y = y_real.copy(deep=True)
        y = y[~X.duplicated(keep='first')]
        X = X[~X.duplicated(keep='first')]
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        for col in ['mileage', 'engine', 'max_power']:
            X[col]  = self.extract_number(X[col])
        X['max_torque_rpm'] = X['torque'].apply(self.extract_rpm)
        X['torque'] = X['torque'].apply(self.extract_torque)

        self.cols_with_na = [i[0] for i in filter(lambda x: x[1] > 0, dict(X.isna().sum()).items())]
        self.train_medians = {}
        for col in self.cols_with_na:
            self.train_medians[col] = X[col].median()
            X[col] = X[col].fillna(self.train_medians[col])

        X['engine'] = X['engine'].apply(int)
        X['seats'] = X['seats'].apply(int)
        X['name'] = X['name'].apply(lambda x: ' '.join(x.split()[:2]))
        X['year'] = X['year'] ** 2
        X['enigne_over_power'] = X['max_power'] / X['engine']
        X['fuel_spent'] = X['km_driven'] / 100 * X['mileage']
        X_y = pd.concat([X, pd.DataFrame(y, columns=['selling_price'])], axis=1)
        self.means = X_y.groupby('name')['selling_price'].mean()
        self.train_mean_y = np.mean(y)
        X['model_avg_price']  = X['name'].map(self.means).fillna(self.train_mean_y)
        X['country'] = X['name'].apply(lambda x: self.get_country(x))

        self.cat_columns = ['fuel', 'seller_type', 'transmission', 'owner', 'country']
        self.ohc_cols = {}
        for col in self.cat_columns: 
            ohc = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
            ohc.fit(X[[col]])
            self.ohc_cols[col] = ohc

        return self

    def transform(self, X_real):
        X = X_real.copy(deep=True)
        skip_idx = X[X.isna().sum(axis=1) > 0].index
        X['skipped_flag'] = X.index.isin(skip_idx)
        for col in ['mileage', 'engine', 'max_power']:
            X[col]  = self.extract_number(X[col])
        X['max_torque_rpm'] = X['torque'].apply(self.extract_rpm)
        X['torque'] = X['torque'].apply(self.extract_torque)
        for col in self.cols_with_na:
            X[col] = X[col].fillna(self.train_medians[col])
        X['engine'] = X['engine'].apply(int)
        X['seats'] = X['seats'].apply(int)
        X['name'] = X['name'].apply(lambda x: ' '.join(x.split()[:2]))
        X['year'] = X['year'] ** 2
        X['enigne_over_power'] = X['max_power'] / X['engine']
        X['fuel_spent'] = X['km_driven'] / 100 * X['mileage']
        X['model_avg_price']  = X['name'].map(self.means).fillna(self.train_mean_y)
        X['country'] = X['name'].apply(lambda x: self.get_country(x))
        X = self.add_ohe_features(X)
        X = X.drop(columns=self.cat_columns)
        X['log_km_driven'] = np.log1p(X['km_driven'])
        X.drop(columns=['km_driven', 'name'], inplace=True)

        return X
    
    def extract_number(self, series):
            return series.apply(
                lambda x: float(re.findall(self.regex_pattern, str(x))[0]) if pd.notnull(x) and re.findall(self.regex_pattern, str(x)) else np.nan
            )
    def extract_rpm(self, x):
        if pd.isna(x):
            return np.nan
        
        match = re.findall(r"(@|at|/)\s*[^0-9]*(?:[0-9,.]+-)?([0-9,.]*)", str(x), re.IGNORECASE)
        if not match:
            return np.nan
        match = float(str(match[0][1]).replace(',', ''))

        return match

    def extract_torque(self, x):
        if pd.isna(x):
            return np.nan
        # регекс подобран методом проб и ошибок
        match = re.findall(r"([-+]?(?:\d*\.\d+|\d+))[^a-zA-Z]*(kgm|nm)?", str(x), re.IGNORECASE)[0]
        if not match:
            return np.nan

        unit = match[1].lower() if match[1] else None
        number = float(match[0]) * (1 if unit != 'kgm' else 9.8)
        return number

    def get_country(self, name):
        # P.S. воспользовался chatgpt для того, чтобы сформировать этот словарь
        manufacturer_country = {
        "Maruti": "India",
        "Skoda": "Czech Republic",
        "Hyundai": "South Korea",
        "Toyota": "Japan",
        "Ford": "USA",
        "Renault": "France",
        "Mahindra": "India",
        "Honda": "Japan",
        "Chevrolet": "USA",
        "Fiat": "Italy",
        "Datsun": "Japan",
        "Tata": "India",
        "Jeep": "USA",
        "Mercedes-Benz": "Germany",
        "Mitsubishi": "Japan",
        "Audi": "Germany",
        "Volkswagen": "Germany",
        "BMW": "Germany",
        "Nissan": "Japan",
        "Lexus": "Japan",
        "Jaguar": "United Kingdom",
        "Land": "United Kingdom",
        "MG": "United Kingdom",
        "Volvo": "Sweden",
        "Daewoo": "South Korea",
        "Kia": "South Korea",
        "Force": "India",
        "Ambassador": "India",
        "Isuzu": "Japan",
        "Peugeot": "France",
        "Opel": "Germany",
        "Ashok": "India"
    }
        return manufacturer_country[name.split()[0]]
    
    def add_ohe_features(self, X):
        for col in self.cat_columns:
            ohc = self.ohc_cols[col]
            X_encoded = ohc.transform(X[[col]])
            X[ohc.get_feature_names_out([col])] = X_encoded
        return X

st.set_page_config(
    page_title="Churn Prediction",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource  # Кэшируем модель (загружается только один раз)
def load_model():
    with open('models/ridge_new_features.pkl', 'rb') as f:
        model = pickle.load(f)
        feature_names = model.named_steps['transformer'].feature_names_
    return model, feature_names

@st.cache_data  # Кэшируем загруженные данные
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file).drop(columns='Unnamed: 0')

# Загружаем модель
model, feature_names = load_model()
model_coef = model.named_steps['model'].coef_
model_cat_columns = model.named_steps['transformer'].cat_columns_unique
model_num_columns = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque_NM', 'torque_rpm']

one_df_checkbox = False

# --- Визуализации ---
st.subheader("📶 Визуализации")

vis_file = st.file_uploader("Загрузите CSV для визуализации", type=["csv"])
if vis_file:
    with st.expander("Развернуть"):
        vis_df = load_data(vis_file)
        
        # анализ целевой переменной (если доступна)
        
        try:
            # st.subheader("Распределение целевой переменной")
            with st.expander("Распределение целевой переменной"):
                fig1 = px.histogram(vis_df, x='selling_price', nbins=30)
                st.plotly_chart(fig1, use_container_width=True)
        except:
            pass

        # парные графики
        with st.expander("Парные графики"):
            fig2 = sns.pairplot(vis_df)
            st.pyplot(fig2)

        # phik корреляция
        with st.expander("phik корреляция"):
            corr_phik = vis_df.drop(columns=['name']).phik_matrix()
            fig3, axes3 = plt.subplots(1, 1, figsize=(20, 8))
            sns.heatmap(corr_phik, annot=True, square=True, fmt='.2f', cmap='coolwarm', center=0, ax=axes3)
            st.pyplot(fig3)




# --- Форма для загрузки файла для предсказания ---

st.subheader("📈 Предсказание на основе датасета")
if vis_file:
    one_df_checkbox = st.checkbox("Испльзовать датасет визуализации")
# Загрузка файла
if one_df_checkbox:
    uploaded_file = vis_file
else:
    uploaded_file = st.file_uploader("Загрузите CSV для предсказания", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)
    # print(feature_names, len(feature_names), len(model_coef))
    try:
        y_true = df['selling_price']
        df.drop(columns='selling_price', inplace=True)
        y_pred = model.predict(df)
    except Exception:
        y_pred = model.predict(df)
        y_true = None
    show_df = df.copy(deep=True)
    if y_true is not None:
        show_df['real_price'] = y_true
    show_df['prediction'] = y_pred.astype(int)
    st.dataframe(show_df)
    

# --- Форма для предсказания ---
st.subheader("🔮 Сделать предсказание для новой машины")

with st.form("prediction_form"):
    col_left, col_right = st.columns(2)
    input_data = {}
    
    with col_left:
        st.write("**Категориальные:**")
        for col in model_cat_columns:
            unique_vals = sorted(model_cat_columns[col].astype(str).tolist())
            input_data[col] = st.selectbox(col, unique_vals, key=f"cat_{col}")
    
    with col_right:
        st.write("**Числовые:**")
        for col in model_num_columns:
            input_data[col] = st.number_input(col, 1, key=f"cat_{col}")
        input_data['mileage'] = f"{input_data['mileage']} kmpl"
        input_data['engine'] = f"{input_data['engine']} CC"
        input_data['max_power'] = f"{input_data['max_power']} bhp"
        input_data['torque'] = f"{input_data['torque_NM']}NM @ 0-{input_data['torque_rpm']} rpm"
        # input_data['torque'] = "172Nm@ 4300rpm"
        del input_data['torque_NM']
        del input_data['torque_rpm']
        input_data


    submitted = st.form_submit_button("Предсказать", use_container_width=True)

if submitted:
    try:
        input_df = pd.DataFrame([input_data])[['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission',
       'owner', 'mileage', 'engine', 'max_power', 'torque', 'seats']]
        pred_y = model.predict(input_df)[0]
        st.success(f"**Оценка:** машина будет стоить {pred_y:.4f} рублей")
        # st.progress(pred_y, text=f"Вероятность оттока: {prob:.1%}")
    except Exception as e:
        st.error(f"❌ Ошибка при предсказании: {e}")


# --- Визуализация весов модели ---
st.subheader("📊 Визуализация весов модели")

feature_importance = pd.DataFrame({
    'feature': feature_names,
    'coefficients': model_coef,
    'abs_coef': abs(model_coef)
}).sort_values('abs_coef', ascending=False)
fig4, ax = plt.subplots(figsize=(8, 10))
sns.barplot(data=feature_importance, x='coefficients', y='feature', orient='h', palette='coolwarm', ax=ax)
st.pyplot(fig4)