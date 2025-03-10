import streamlit as st
from buoi5.Clustering_Algorithms import ClusteringAlgorithms

if "last_page" in st.session_state and st.session_state.last_page != "clustering":
    st.session_state.clear()  # Xóa toàn bộ session

st.session_state.last_page = "clustering" 


st.title("🔍 Clustering Algorithms")


# Gọi hàm ClusteringAlgorithms từ module
ClusteringAlgorithms()
