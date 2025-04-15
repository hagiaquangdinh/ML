import streamlit as st
import os
import cv2
import requests
import io
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import random
import struct
from sklearn.datasets import load_iris
import mlflow
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from PIL import Image
from sklearn.model_selection import KFold
from collections import Counter
from mlflow.tracking import MlflowClient
# Khởi tạo mô hình Logistic Regression
from sklearn.linear_model import LogisticRegression

def load_data():
    X_text = np.load("Data/alphabet_X.npy")
    y_text = np.load("Data/alphabet_y.npy")
    X_geometric = np.load("Data/geometric_X.npy")
    y_geometric = np.load("Data/geometric_y.npy")
    return X_text, y_text, X_geometric, y_geometric




def run_ClassificationMinst_app():
    # Load default data
    X_text, y_text, X_geometric, y_geometric = load_data()
    
    # For training, we can use a portion of X_text or combine datasets; here we split X_text
    X_text, X_geometric, y_text, y_geometric = train_test_split(X_text, y_text, test_size=0.2, random_state=42)
    
    # Store in session_state for access across tabs
    st.session_state["X_test"] = X_text
    st.session_state["y_test"] = y_text
    st.session_state["X_temp"] = X_geometric
    st.session_state["y_temp"] = y_geometric

    # mlflow_tracking_uri = st.secrets["MLFLOW_TRACKING_URI"]
    # mlflow_username = st.secrets["MLFLOW_TRACKING_USERNAME"]
    # mlflow_password = st.secrets["MLFLOW_TRACKING_PASSWORD"]
    
    # # Thiết lập biến môi trường
    # os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
    # os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_username
    # os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_password
    
    # # Thiết lập MLflow (Đặt sau khi mlflow_tracking_uri đã có giá trị)
    # mlflow.set_tracking_uri(mlflow_tracking_uri)

    # # Định nghĩa đường dẫn đến các file MNIST
    # dataset_path = os.path.dirname(os.path.abspath(__file__)) 
    # train_images_path = os.path.join(dataset_path, "train-images.idx3-ubyte")
    # train_labels_path = os.path.join(dataset_path, "train-labels.idx1-ubyte")
    # test_images_path = os.path.join(dataset_path, "t10k-images.idx3-ubyte")
    # test_labels_path = os.path.join(dataset_path, "t10k-labels.idx1-ubyte")

    # # Tải dữ liệu
    # train_images = load_mnist_images(train_images_path)
    # train_labels = load_mnist_labels(train_labels_path)
    # test_images = load_mnist_images(test_images_path)
    # test_labels = load_mnist_labels(test_labels_path)

    # Giao diện Streamlit
    st.title("My Application Name")
    tabs = st.tabs([
        "Tiền Xử lí dữ liệu",
        "Huấn luyện",
        "Dự đoán",
        "Mlflow",
    ])
    # tab_info, tab_load, tab_preprocess, tab_split,  tab_demo, tab_log_info = tabs
    tab_load, tab_preprocess,  tab_demo ,tab_mlflow= tabs

   
    with tab_load:
        st.subheader("Chọn dữ liệu đầu vào")
        data_option = st.radio(
            "Chọn loại dữ liệu:",
            ("Dữ liệu chữ cái (Text)", "Dữ liệu hình học (Geometric)", "Tải lên từ file CSV")
        )

        if data_option == "Dữ liệu chữ cái (Text)":
            st.info("Đang sử dụng dữ liệu chữ cái.")
            X = X_text
            y = y_text
            st.session_state["uploaded_data_type"] = "text"
            st.session_state["X_original"] = X
            st.session_state["y_original"] = y
            st.success(f"✅ Đã tải dữ liệu chữ cái với kích thước: {X.shape}")
        elif data_option == "Dữ liệu hình học (Geometric)":
            st.info("Đang sử dụng dữ liệu hình học.")
            X = X_geometric
            y = y_geometric
            st.session_state["uploaded_data_type"] = "geometric"
            st.session_state["X_original"] = X
            st.session_state["y_original"] = y
            st.success(f"✅ Đã tải dữ liệu hình học với kích thước: {X.shape}")
        elif data_option == "Tải lên từ file CSV":
            uploaded_file = st.file_uploader("📂 Chọn file để tải lên ", type=["csv", "txt"])
            if uploaded_file is not None:
                try:
                    # Đọc file CSV
                    data = pd.read_csv(uploaded_file)
                    
                    # Giả sử cột cuối cùng là nhãn (y), các cột còn lại là đặc trưng (X)
                    X = data.iloc[:, :-1].values  # Đặc trưng
                    y = data.iloc[:, -1].values   # Nhãn
                    
                    # Lưu dữ liệu vào session_state
                    st.session_state["X"] = X
                    st.session_state["y"] = y
                    st.success("✅ Dữ liệu đã được tải thành công từ file!")
                    
                    # Hiển thị dữ liệu
                    st.write("**Dữ liệu:**")
                    st.dataframe(data.head())
                except Exception as e:
                    st.error(f"🚨 Lỗi khi đọc file CSV: {e}")
                



    # 3️⃣ HUẤN LUYỆN MÔ HÌNH
    with tab_preprocess:
        with st.expander("**Phân chia dữ liệu**", expanded=True):    
            if "X" in st.session_state and "y" in st.session_state:
                X = st.session_state["X"]
                y = st.session_state["y"]

                # Reshape nếu cần (giả sử dữ liệu không phải ảnh, không cần reshape như MNIST)
                if len(X.shape) > 2:  # Nếu dữ liệu có dạng ảnh (3D)
                    X = X.reshape(X.shape[0], -1)

                # Sliders cho tỷ lệ phân chia
                st.write("🔹 Chọn tỷ lệ cho các tập dữ liệu (tổng phải bằng 100%):")
                train_size = st.slider(
                    "Tỷ lệ tập huấn luyện (%)",
                    min_value=10, max_value=80, value=60, step=5
                ) / 100
                max_val_size = 1.0 - train_size - 0.1
                val_size = st.slider(
                    "Tỷ lệ tập validation (%)",
                    min_value=10, max_value=int(max_val_size * 100), value=20, step=5
                ) / 100
                test_size = 1.0 - train_size - val_size

                st.write(f"Tỷ lệ tập kiểm tra (test): {test_size * 100:.0f}%")
                if abs(train_size + val_size + test_size - 1.0) > 0.01:
                    st.error("🚨 Tổng tỷ lệ không bằng 100%. Vui lòng điều chỉnh lại!")
                else:
                    # Phân chia dữ liệu
                    X_temp, X_test, y_temp, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )
                    val_size_adjusted = val_size / (train_size + val_size)
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_temp, y_temp, test_size=val_size_adjusted, random_state=42
                    )

                    # Hiển thị kết quả phân chia
                    total_samples = X.shape[0]
                    train_percent = (X_train.shape[0] / total_samples) * 100
                    val_percent = (X_val.shape[0] / total_samples) * 100
                    test_percent = (X_test.shape[0] / total_samples) * 100


                    st.write(f"📊 **Tỷ lệ phân chia thực tế**: Train={train_percent:.0f}%, Validation={val_percent:.0f}%, Test={test_percent:.0f}%")
                    st.write("✅ Dữ liệu đã được xử lý và chia tách.")
                    st.write(f"🔹 Kích thước tập huấn luyện: `{X_train.shape}`")
                    st.write(f"🔹 Kích thước tập validation: `{X_val.shape}`")
                    st.write(f"🔹 Kích thước tập kiểm tra: `{X_test.shape}`")

                    # Lưu dữ liệu đã phân chia vào session_state
                    st.session_state["X_train"] = X_train
                    st.session_state["y_train"] = y_train
                    st.session_state["X_val"] = X_val
                    st.session_state["y_val"] = y_val
                    st.session_state["X_test"] = X_test
                    st.session_state["y_test"] = y_test
            else:
                st.error("🚨 Dữ liệu chưa được nạp. Hãy tải dữ liệu trước từ tab 'Tiền Xử lý dữ liệu'.")

        with st.expander("**Huấn luyện mô hình**", expanded=True):
            if "X_train" in st.session_state:
                X_train = st.session_state["X_train"]
                y_train = st.session_state["y_train"]
                X_val = st.session_state["X_val"]
                y_val = st.session_state["y_val"]

                model_option = st.radio("🔹 Chọn mô hình huấn luyện:", ("Decision Tree", "SVM"))
                if model_option == "Decision Tree":
                    st.subheader("🌳 Decision Tree Classifier")
                    max_depth = st.slider("Chọn độ sâu tối đa của cây:", min_value=1, max_value=20, value=5)
                    n_folds = st.slider("Chọn số folds cho K-Fold Cross-Validation:", min_value=2, max_value=10, value=5)

                    if st.button("🚀 Huấn luyện mô hình"):
                        with st.spinner("Đang huấn luyện mô hình..."):
                            with mlflow.start_run():
                                dt_model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
                                kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
                                cv_scores = []

                                progress_bar = st.progress(0)
                                progress_text = st.empty()
                                total_folds = n_folds

                                for i, (train_index, val_index) in enumerate(kf.split(X_train)):
                                    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
                                    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
                                    dt_model.fit(X_train_fold, y_train_fold)
                                    y_val_pred_fold = dt_model.predict(X_val_fold)
                                    fold_accuracy = accuracy_score(y_val_fold, y_val_pred_fold)
                                    cv_scores.append(fold_accuracy)

                                    progress = (i + 1) / total_folds
                                    progress_bar.progress(progress)
                                    progress_text.text(f"Tiến trình huấn luyện: {int(progress * 100)}%")

                                mean_cv_accuracy = np.mean(cv_scores)
                                std_cv_accuracy = np.std(cv_scores)

                                dt_model.fit(X_train, y_train)
                                y_val_pred_dt = dt_model.predict(X_val)
                                accuracy_dt = accuracy_score(y_val, y_val_pred_dt)

                                mlflow.log_param("model_type", "Decision Tree")
                                mlflow.log_param("max_depth", max_depth)
                                mlflow.log_param("n_folds", n_folds)
                                mlflow.log_metric("mean_cv_accuracy", mean_cv_accuracy)
                                mlflow.log_metric("std_cv_accuracy", std_cv_accuracy)
                                mlflow.log_metric("accuracy", accuracy_dt)
                                mlflow.sklearn.log_model(dt_model, "decision_tree_model")

                                st.session_state["selected_model_type"] = "Decision Tree"
                                st.session_state["trained_model"] = dt_model 
                                st.session_state["X_train"] = X_train 
                                st.session_state["dt_max_depth"] = max_depth
                                st.session_state["n_folds"] = n_folds 

                                st.markdown("---") 
                                st.write(f"🔹Mô hình được chọn để đánh giá: `{model_option}`")
                                st.write("🔹 Tham số mô hình:")
                                # personalised_write(f"- **Độ sâu tối đa**: `{max_depth}`")
                                st.write(f"- **Số folds trong Cross-Validation**: `{n_folds}`")
                                st.write(f"✅ **Độ chính xác trung bình từ K-Fold Cross-Validation ({n_folds} folds):** `{mean_cv_accuracy:.4f} ± {std_cv_accuracy:.4f}`")
                                st.write(f"✅ **Độ chính xác trên tập validation:** `{accuracy_dt:.4f}`")
                            mlflow.end_run()
                # Thêm logic tương tự cho các mô hình khác (SVM, Logistic Regression) nếu cần
            else:
                st.error("🚨 Dữ liệu chưa được phân chia. Hãy thực hiện phân chia dữ liệu trước.")
        

    with tab_demo:   
        with st.expander("**Dự đoán kết quả**", expanded=True):
            st.write("**Dự đoán trên ảnh do người dùng tải lên**")

            # Kiểm tra xem mô hình đã được huấn luyện và lưu kết quả chưa
            if "selected_model_type" not in st.session_state or "trained_model" not in st.session_state:
                st.warning("⚠️ Chưa có mô hình nào được huấn luyện. Vui lòng huấn luyện mô hình trước khi dự đoán.")
            else:
                best_model_name = st.session_state.selected_model_type
                best_model = st.session_state.trained_model

                st.write(f"🎯 Mô hình đang sử dụng: `{best_model_name}`")
                # st.write(f"✅ Độ chính xác trên tập kiểm tra: `{st.session_state.get('test_accuracy', 'N/A'):.4f}`")

                # Lấy các tham số từ session_state để hiển thị
                if best_model_name == "Decision Tree":
                    criterion = st.session_state.get("dt_criterion", "entropy")
                    max_depth = st.session_state.get("dt_max_depth", 5)  # Giá trị mặc định là 5
                    n_folds = st.session_state.get("n_folds", 5)  # Giá trị mặc định là 5
                    st.write("🔹 **Tham số mô hình Decision Tree:**")
                    st.write(f"- **Tiêu chí phân nhánh**: `{criterion}`")
                    st.write(f"- **Độ sâu tối đa**: `{max_depth}`")
                    st.write(f"- **Số folds trong Cross-Validation**: `{n_folds}`")
                elif best_model_name == "Logistic Regression":
                    C = st.session_state.get("lr_C", 1.0)
                    n_folds = st.session_state.get("n_folds", 5)
                    st.write("🔹 **Tham số mô hình Logistic Regression:**")
                    st.write(f"- **C (Regularization)**: `{C}`")
                    st.write(f"- **Số folds trong Cross-Validation**: `{n_folds}`")

                # Cho phép người dùng tải lên ảnh
                uploaded_file = st.file_uploader("📂 Chọn một ảnh để dự đoán", type=["png", "jpg", "jpeg"])

                if uploaded_file is not None:
                    # Đọc ảnh từ tệp tải lên
                    image = Image.open(uploaded_file).convert("L")  # Chuyển sang ảnh xám
                    image = np.array(image)

                    # Kiểm tra xem dữ liệu huấn luyện đã lưu trong session_state hay chưa
                    if "X_train" in st.session_state:
                        X_train_shape = st.session_state["X_train"].shape[1]  # Lấy số đặc trưng từ tập huấn luyện

                        # Resize ảnh về kích thước phù hợp với mô hình đã huấn luyện
                        image = cv2.resize(image, (28, 28))  # Cập nhật kích thước theo dữ liệu ban đầu
                        image = image.reshape(1, -1)  # Chuyển về vector 1 chiều

                        # Đảm bảo số chiều đúng với dữ liệu huấn luyện
                        if image.shape[1] == X_train_shape:
                            prediction = best_model.predict(image)[0]

                            # Hiển thị ảnh và kết quả dự đoán
                            st.image(uploaded_file, caption="📷 Ảnh bạn đã tải lên", use_container_width=True)
                            st.success(f"✅ **Dự đoán:** {prediction}")
                        else:
                            st.error(f"🚨 Ảnh không có số đặc trưng đúng ({image.shape[1]} thay vì {X_train_shape}). Hãy kiểm tra lại dữ liệu đầu vào!")
                    else:
                        st.error("🚨 Dữ liệu huấn luyện không tìm thấy. Hãy huấn luyện mô hình trước khi dự đoán.")

    with tab_mlflow:
        st.header("Thông tin Huấn luyện & MLflow UI")
        try:
            client = MlflowClient()
            experiment_name = "Application"
    
            # Kiểm tra nếu experiment đã tồn tại
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = client.create_experiment(experiment_name)
                st.success(f"Experiment mới được tạo với ID: {experiment_id}")
            else:
                experiment_id = experiment.experiment_id
                st.info(f"Đang sử dụng experiment ID: {experiment_id}")
    
            mlflow.set_experiment(experiment_name)
    
    
            # Truy vấn các run trong experiment
            runs = client.search_runs(experiment_ids=[experiment_id])
    
            # 1) Chọn và đổi tên Run Name
            st.subheader("Đổi tên Run")
            if runs:
                run_options = {run.info.run_id: f"{run.data.tags.get('mlflow.runName', 'Unnamed')} - {run.info.run_id}"
                               for run in runs}
                selected_run_id_for_rename = st.selectbox("Chọn Run để đổi tên:", 
                                                          options=list(run_options.keys()), 
                                                          format_func=lambda x: run_options[x])
                new_run_name = st.text_input("Nhập tên mới cho Run:", 
                                             value=run_options[selected_run_id_for_rename].split(" - ")[0])
                if st.button("Cập nhật tên Run"):
                    if new_run_name.strip():
                        client.set_tag(selected_run_id_for_rename, "mlflow.runName", new_run_name.strip())
                        st.success(f"Đã cập nhật tên Run thành: {new_run_name.strip()}")
                    else:
                        st.warning("Vui lòng nhập tên mới cho Run.")
            else:
                st.info("Chưa có Run nào được log.")
    
            # 2) Xóa Run
            st.subheader("Danh sách Run")
            if runs:
                selected_run_id_to_delete = st.selectbox("", 
                                                         options=list(run_options.keys()), 
                                                         format_func=lambda x: run_options[x])
                if st.button("Xóa Run", key="delete_run"):
                    client.delete_run(selected_run_id_to_delete)
                    st.success(f"Đã xóa Run {run_options[selected_run_id_to_delete]} thành công!")
                    st.experimental_rerun()  # Tự động làm mới giao diện
            else:
                st.info("Chưa có Run nào để xóa.")
    
            # 3) Danh sách các thí nghiệm
            st.subheader("Danh sách các Run đã log")
            if runs:
                selected_run_id = st.selectbox("Chọn Run để xem chi tiết:", 
                                               options=list(run_options.keys()), 
                                               format_func=lambda x: run_options[x])
    
                # 4) Hiển thị thông tin chi tiết của Run được chọn
                selected_run = client.get_run(selected_run_id)
                st.write(f"**Run ID:** {selected_run_id}")
                st.write(f"**Run Name:** {selected_run.data.tags.get('mlflow.runName', 'Unnamed')}")
    
                st.markdown("### Tham số đã log")
                st.json(selected_run.data.params)
    
                st.markdown("### Chỉ số đã log")
                metrics = {
                    "mean_cv_accuracy": selected_run.data.metrics.get("mean_cv_accuracy", "N/A"),
                    "std_cv_accuracy": selected_run.data.metrics.get("std_cv_accuracy", "N/A"),
                    "accuracy": selected_run.data.metrics.get("accuracy", "N/A"),
                    "model_type": selected_run.data.metrics.get("model_type", "N/A"),
                    "kernel": selected_run.data.metrics.get("kernel", "N/A"),
                    "C_value": selected_run.data.metrics.get("C_value", "N/A")
                

                }
                st.json(metrics)
    
                # 5) Nút bấm mở MLflow UI
                st.subheader("Truy cập MLflow UI")
                mlflow_url = "https://dagshub.com/quangdinhhusc/HMVPYTHON.mlflow"
                if st.button("Mở MLflow UI"):
                    st.markdown(f'**[Click để mở MLflow UI]({mlflow_url})**')
            else:
                st.info("Chưa có Run nào được log. Vui lòng huấn luyện mô hình trước.")
    
        except Exception as e:
            st.error(f"Không thể kết nối với MLflow: {e}")




if __name__ == "__main__":
    run_ClassificationMinst_app()
    
