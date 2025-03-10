import streamlit as st
from buoi6.PCA_t_SNE import pca_tsne
if "last_page" in st.session_state and st.session_state.last_page != "pca":
    st.session_state.clear()  # Xóa toàn bộ session

st.session_state.last_page = "pca" 

st.title("🔍 Thuật toán giảm chiều")


# Gọi hàm ClusteringAlgorithms từ module
pca_tsne()
