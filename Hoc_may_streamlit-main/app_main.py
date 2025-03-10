import streamlit as st

st.set_page_config(
    page_title="Multi-Page App",
    page_icon="📊",
    layout="wide",
)

st.sidebar.title("🏠 Home")
st.sidebar.write("Chọn ứng dụng từ sidebar!")

st.title("🎯 Welcome to Multi-Page Streamlit App")
st.write("👉 Sử dụng sidebar để chọn ứng dụng bạn muốn chạy.")

st.write("📌 Ứng dụng bao gồm:")
st.markdown("- 📈 **Linear Regression**")
st.markdown("- 🔢 **Classification MNIST**")
st.markdown("- 🔍 **Clustering Algorithms**")
st.markdown("- 🔍 **Chạy thuật toán PCA và t-SNE**")