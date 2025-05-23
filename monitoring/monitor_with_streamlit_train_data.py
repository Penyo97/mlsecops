import streamlit as st
import pandas as pd


st.markdown("# Train Dataset ")
st.sidebar.markdown("# Train Dataset ")


def get_raw_data():
    df_train = pd.read_csv('car_price_80.csv')
    df_test = pd.read_csv('car_price_20.csv')
    experiment_name = '1'

    return df_train, df_test


header = st.container()
dataset = st.container()
plot_price = st.container()
manufacturer = st.container()

with header:
    st.title('Monitoring some elements')
    st.text("You can see some examples of Streamlit'possibilities")

with dataset:
    st.header("Car dataset import")
    st.text("You can see here a sample from the train dataset")

    df_train, df_test = get_raw_data()
    st.write(df_train.head(20))

with plot_price:
    st.header("Price value counts")
    total_day_minutes = df_train["Price"].value_counts().head(50)
    st.bar_chart(total_day_minutes)

with manufacturer:
    st.header("Manufacturer value counts")
    total_day_minutes = df_train["Manufacturer"].value_counts().head(50)
    st.scatter_chart(total_day_minutes)

# python -m streamlit run monitor_with_streamlit_train_data.py