import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import joblib
import pandas as pd



# Load dữ liệu MNIST
def ly_thuyet_Decision_tree():
    st.header("📖 Lý thuyết về Decision Tree")

    st.subheader("1️⃣ Giới thiệu về Decision Tree")
    st.write("""
    - **Decision Tree** hoạt động bằng cách chia nhỏ dữ liệu theo điều kiện để phân loại chính xác.
    - Mỗi nhánh trong cây là một câu hỏi "Có/Không" dựa trên đặc trưng dữ liệu.
    - Mô hình này dễ hiểu và trực quan nhưng có thể bị **overfitting** nếu không giới hạn độ sâu.
    """)

    # Hiển thị ảnh minh họa Decision Tree
    st.image("buoi4/img1.png", caption="Ví dụ về cách Decision Tree phân chia dữ liệu", use_container_width="auto")

    st.subheader("2️⃣ Các bước thực hiện trong Decision Tree")
    st.write("""
    **Bước 1: Tính Entropy của tập dữ liệu ban đầu**
    - Entropy đo lường mức độ hỗn loạn của dữ liệu. Nếu dữ liệu hoàn toàn đồng nhất, Entropy = 0.
    - Công thức Entropy:
    """)
    st.latex(r"""
    H(S) = - \sum_{i=1}^{c} p_i \log_2 p_i
    """)
    st.write(r"""
    Trong đó:
    - $$ c $$ : số lượng lớp trong tập dữ liệu.
    - $$p_i $$ : xác suất xuất hiện của lớp $$ i $$, được tính bằng tỷ lệ số mẫu của lớp $$ i $$trên tổng số mẫu.
    """)

    st.write(r"""
    **Bước 2:  Tính Entropy \( $$H(S_j)$$ \) của từng tập con khi chia theo từng thuộc tính**
    - Mỗi lần chia, dữ liệu được tách thành nhiều tập con nhỏ hơn.
    - Entropy của mỗi tập con được tính tương tự như Entropy ban đầu.
    """)
    
    st.write("""
    **Bước 3: Tính Information Gain (IG) của từng thuộc tính**
    - IG đo lường mức độ giảm Entropy khi chia dữ liệu theo một thuộc tính.
    - Công thức Information Gain:
    """)
    st.latex(r"""
    IG = H(S) - \sum_{j=1}^{k} \frac{|S_j|}{|S|} H(S_j)
    """)
    st.write(r"""
    Trong đó:
    - $$ S $$: tập dữ liệu ban đầu.
    - $$S_j $$: tập con sau khi chia theo thuộc tính đang xét.
    - $$ \frac{|S_j|}{|S|} $$ : tỷ lệ số lượng mẫu trong tập con $$ S_j $$ so với tổng số mẫu.
    - $$H(S) $$ : Entropy của tập dữ liệu ban đầu.
    - $$ H(S_j) $$ : Entropy của tập con $$S_j $$.
    """)
    
    st.write("""
    **Bước 4: Chọn thuộc tính có Information Gain cao nhất để phân nhánh**
    - Thuộc tính có IG cao nhất sẽ được chọn để chia tập dữ liệu.
    """)
    
    st.write("""
    **Bước 5: Lặp lại quá trình trên cho từng nhánh của cây**
    - Quá trình chia nhỏ tiếp tục đến khi các tập con không thể chia nhỏ hơn hoặc đạt điều kiện dừng.
    """)


    
    
    
def ly_thuyet_SVM():
    st.title("🔎 Support Vector Machine (SVM)")
    
    # 1️⃣ Tổng quan về SVM
    st.header("1️⃣ Tổng quan về SVM")
    st.write(r"""
    - **Support Vector Machine (SVM)** là một thuật toán học máy mạnh mẽ dùng để phân loại dữ liệu.
    - **Mục tiêu chính**: Tìm **siêu phẳng (hyperplane) tối ưu** để phân tách các lớp dữ liệu.
    - **Ứng dụng**: Nhận diện khuôn mặt, phát hiện thư rác, phân loại văn bản, nhận dạng chữ viết tay,...
    - **Ưu điểm**:
      ✅ Hiệu quả trên dữ liệu có độ nhiễu thấp.
      ✅ Có thể phân loại dữ liệu không tuyến tính bằng **Kernel Trick**.
    - **Nhược điểm**:
      ❌ Chậm trên tập dữ liệu lớn do tính toán phức tạp.
      ❌ Nhạy cảm với tham số $$C$$và lựa chọn Kernel.
    """)
    
    st.image("buoi4/img2.png", use_container_width=True, caption="SVM tìm siêu phẳng tối ưu để phân tách dữ liệu")
    
    # 2️⃣ Cách hoạt động của SVM
    st.header("2️⃣ Cách hoạt động của SVM")
    st.write("""
    🔹 **Bước 1: Biểu diễn dữ liệu trong không gian nhiều chiều**
    - Dữ liệu được ánh xạ vào một không gian có nhiều chiều hơn, nơi mà có thể tìm được một siêu phẳng để phân tách dữ liệu.

    🔹 **Bước 2: Tìm siêu phẳng tối ưu**
    - Mô hình tìm một siêu phẳng sao cho khoảng cách từ siêu phẳng đến các điểm dữ liệu gần nhất (**support vectors**) là lớn nhất.
    - Nếu dữ liệu **không thể phân tách tuyến tính**, có hai hướng giải quyết:
      ✅ **Dùng Kernel Trick** để ánh xạ dữ liệu sang không gian cao hơn.
      ✅ **Thêm Soft Margin** để chấp nhận một số điểm bị phân loại sai.
    """)
    
    # 3️⃣ Công thức toán học trong SVM
    st.header("3️⃣ Công thức toán học trong SVM")
    
    st.subheader("📌 Tìm siêu phẳng tối ưu")
    st.latex(r"""
    \min_{w, b} \frac{1}{2} ||w||^2
    """)
    
    st.write(r"Mục tiêu là tìm vector trọng số \( w \) nhỏ nhất để tăng khả năng tổng quát hóa mô hình.")
    
    st.subheader("📌 Ràng buộc đảm bảo phân loại đúng")
    st.latex(r"""
    y_i (w \cdot x_i + b) \geq 1, \forall i
    """)
    
    st.write("Mọi điểm dữ liệu phải nằm đúng phía của siêu phẳng để đảm bảo phân loại chính xác.")
    
    st.subheader("📌 Khoảng cách từ một điểm đến siêu phẳng")
    st.latex(r"""
    d = \frac{|w \cdot x + b|}{||w||}
    """)
    
    # 4️⃣ SVM với Soft Margin và biến slack
    st.header("4️⃣ SVM với Soft Margin và biến slack")
    st.write("""
    🔹 Nếu dữ liệu **không thể phân tách hoàn hảo**, ta sử dụng Soft Margin để cho phép một số điểm nằm sai bên lề.
    """)
    
    st.subheader("📌 Hàm mất mát với Soft Margin")
    st.latex(r"""
    \min_{w, b} \frac{1}{2} ||w||^2 + C \sum_{i=1}^{n} \xi_i
    """)
    
    st.write(r"""
    - Thêm biến slack \( \xi_i \) để cho phép một số điểm bị phân loại sai.
    - Ý nghĩa của biến slack:
      - $$ \xi_i = 0 $$: Điểm nằm ngoài hoặc trên lề, được phân loại đúng.
      - $$0 < \xi_i < 1 $$: Điểm nằm trong lề nhưng vẫn được phân loại đúng.
      - $$ \xi_i > 1 $$ : Điểm bị phân loại sai.
    """)
    
    st.write(r"""
    📍 **Ý nghĩa của hệ số $$ C $$**
    - Nếu $$ C $$lớn → Mô hình cố gắng phân loại chính xác nhất có thể nhưng dễ bị **overfitting**.
    - Nếu $$C $$ nhỏ → Mô hình linh hoạt hơn nhưng có thể chấp nhận nhiều lỗi hơn.
    """)
    
    # 5️⃣ Tổng kết
    st.header("5️⃣ Tổng kết")
    st.write(r"""
    ✅ **SVM tìm kiếm siêu phẳng tối ưu** để phân tách dữ liệu, đảm bảo khoảng cách giữa hai lớp là lớn nhất.
    ✅ **Nếu dữ liệu không tuyến tính**, SVM sử dụng **Kernel Trick** để ánh xạ dữ liệu sang không gian cao hơn.
    ✅ **Nếu dữ liệu có nhiễu**, SVM sử dụng **Soft Margin** để chấp nhận một số điểm bị phân loại sai.
    ✅ **Tham số quan trọng:**
    - **$$C $$**: Điều chỉnh giữa việc tối ưu margin và chấp nhận lỗi.
    - **Kernel**: Biến đổi dữ liệu để làm việc với dữ liệu không tuyến tính.
    """)

# Gọi hàm hiển thị nội dung lý thuyết về SVM




def data():
    st.header("MNIST Dataset")
    st.write("""
      **MNIST** là một trong những bộ dữ liệu nổi tiếng và phổ biến nhất trong cộng đồng học máy, 
      đặc biệt là trong các nghiên cứu về nhận diện mẫu và phân loại hình ảnh.
  
      - Bộ dữ liệu bao gồm tổng cộng **70.000 ảnh chữ số viết tay** từ **0** đến **9**, 
        mỗi ảnh có kích thước **28 x 28 pixel**.
      - Chia thành:
        - **Training set**: 60.000 ảnh để huấn luyện.
        - **Test set**: 10.000 ảnh để kiểm tra.
      - Mỗi hình ảnh là một chữ số viết tay, được chuẩn hóa và chuyển thành dạng grayscale (đen trắng).
  
      Dữ liệu này được sử dụng rộng rãi để xây dựng các mô hình nhận diện chữ số.
      """)

    st.subheader("Một số hình ảnh từ MNIST Dataset")
    st.image("buoi4/img3.png", caption="Một số hình ảnh từ MNIST Dataset", use_container_width=True)

    st.subheader("Ứng dụng thực tế của MNIST")
    st.write("""
      Bộ dữ liệu MNIST đã được sử dụng trong nhiều ứng dụng nhận dạng chữ số viết tay, chẳng hạn như:
      - Nhận diện số trên các hoá đơn thanh toán, biên lai cửa hàng.
      - Xử lý chữ số trên các bưu kiện gửi qua bưu điện.
      - Ứng dụng trong các hệ thống nhận diện tài liệu tự động.
    """)

    st.subheader("Ví dụ về các mô hình học máy với MNIST")
    st.write("""
      Các mô hình học máy phổ biến đã được huấn luyện với bộ dữ liệu MNIST bao gồm:
      - **Logistic Regression**
      - **Decision Trees**
      - **K-Nearest Neighbors (KNN)**
      - **Support Vector Machines (SVM)**
      - **Convolutional Neural Networks (CNNs)**
    """)

    st.subheader("Kết quả của một số mô hình trên MNIST ")
    st.write("""
      Để đánh giá hiệu quả của các mô hình học máy với MNIST, người ta thường sử dụng độ chính xác (accuracy) trên tập test:
      
      - **Decision Tree**: 0.8574
      - **SVM (Linear)**: 0.9253
      - **SVM (poly)**: 0.9774
      - **SVM (sigmoid)**: 0.7656
      - **SVM (rbf)**: 0.9823
      
      
      
    """)





def plot_tree_metrics():
    # Dữ liệu bạn đã cung cấp

    accuracies = [
        0.4759, 0.5759, 0.6593, 0.7741, 0.8241, 0.8259, 0.8481, 0.8574, 0.8537, 0.8463,
        0.8463, 0.8426, 0.8426, 0.8426, 0.8426, 0.8426, 0.8426, 0.8426
    ]
    tree_depths = [
        3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
    ]

    # Tạo DataFrame từ dữ liệu
    data = pd.DataFrame({
        "Tree Depth": tree_depths,
        "Accuracy": accuracies
    })

    # Vẽ biểu đồ với st.line_chart
    st.subheader("Độ chính xác theo chiều sâu cây quyết định")
    st.line_chart(data.set_index('Tree Depth'))



import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_mnist_data():
    X = np.load("buoi4/X.npy")
    y = np.load("buoi4/y.npy")
    return X, y
def split_data():
    st.title("📌 Chia dữ liệu Train/Test")

    # Đọc dữ liệu
    X, y = load_mnist_data() 
    total_samples = X.shape[0]

    
    # Nếu chưa có cờ "data_split_done", đặt mặc định là False
    if "data_split_done" not in st.session_state:
        st.session_state.data_split_done = False  

    # Thanh kéo chọn số lượng ảnh để train
    num_samples = st.slider("📌 Chọn số lượng ảnh để train:", 1000, total_samples, 10000)
    
    # Thanh kéo chọn tỷ lệ Train/Test
    test_size = st.slider("📌 Chọn % dữ liệu Test", 10, 50, 20)
    remaining_size = 100 - test_size
    val_size = st.slider("📌 Chọn % dữ liệu Validation (trong phần Train)", 0, 50, 15)
    st.write(f"📌 **Tỷ lệ phân chia:** Test={test_size}%, Validation={val_size}%, Train={remaining_size - val_size}%")

    if st.button("✅ Xác nhận & Lưu") and not st.session_state.data_split_done:
        st.session_state.data_split_done = True  # Đánh dấu đã chia dữ liệu
        
        # Chia dữ liệu theo tỷ lệ đã chọn
        X_selected, _, y_selected, _ = train_test_split(
            X, y, train_size=num_samples, stratify=y, random_state=42
        )

        # Chia train/test
        stratify_option = y_selected if len(np.unique(y_selected)) > 1 else None
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X_selected, y_selected, test_size=test_size/100, stratify=stratify_option, random_state=42
        )

        # Chia train/val
        stratify_option = y_train_full if len(np.unique(y_train_full)) > 1 else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=val_size / (100 - test_size),
            stratify=stratify_option, random_state=42
        )

        # Lưu dữ liệu vào session_state
        st.session_state.total_samples= num_samples
        st.session_state.X_train = X_train
        st.session_state.X_val = X_val
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_val = y_val
        st.session_state.y_test = y_test
        st.session_state.test_size = X_test.shape[0]
        st.session_state.val_size = X_val.shape[0]
        st.session_state.train_size = X_train.shape[0]

        # Hiển thị thông tin chia dữ liệu
        summary_df = pd.DataFrame({
            "Tập dữ liệu": ["Train", "Validation", "Test"],
            "Số lượng mẫu": [X_train.shape[0], X_val.shape[0], X_test.shape[0]]
        })
        st.success("✅ Dữ liệu đã được chia thành công!")
        st.table(summary_df)

    elif st.session_state.data_split_done:
        st.info("✅ Dữ liệu đã được chia, không cần chạy lại.")

        
        
        
        
import os
import mlflow
from mlflow.tracking import MlflowClient

    
    
from sklearn.model_selection import cross_val_score

def train():
    #mlflow_input()
    # 📥 **Tải dữ liệu MNIST**
    if "X_train" in st.session_state:
        X_train=st.session_state.X_train 
        X_val=st.session_state.X_val
        X_test=st.session_state.X_test 
        y_train=st.session_state.y_train 
        y_val=st.session_state.y_val 
        y_test=st.session_state.y_test 
    else:
        st.error("⚠️ Chưa có dữ liệu! Hãy chia dữ liệu trước.")
        return

    # 🌟 Chuẩn hóa dữ liệu
    X_train = X_train.reshape(-1, 28 * 28) / 255.0
    X_test = X_test.reshape(-1, 28 * 28) / 255.0

    st.header("⚙️ Chọn mô hình & Huấn luyện")

    # 📌 **Chọn mô hình**
    model_choice = st.selectbox("Chọn mô hình:", ["Decision Tree", "SVM"])

    if model_choice == "Decision Tree":
        st.markdown("""
        - **🌳 Decision Tree (Cây quyết định)** giúp chia dữ liệu thành các nhóm bằng cách đặt câu hỏi nhị phân dựa trên đặc trưng.
        - **Tham số cần chọn:**  
            - **max_depth**: Giới hạn độ sâu tối đa của cây.  
        """)
        max_depth = st.slider("max_depth", 1, 20, 5)
        model = DecisionTreeClassifier(max_depth=max_depth)

    elif model_choice == "SVM":
        st.markdown("""
        - **🛠️ SVM (Support Vector Machine)** là mô hình tìm siêu phẳng tốt nhất để phân tách dữ liệu.
        """)
        C = st.slider("C (Regularization)", 0.1, 10.0, 1.0)
        kernel = st.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])
        model = SVC(C=C, kernel=kernel)
    n_folds = st.slider("Chọn số folds (KFold Cross-Validation):", min_value=2, max_value=10, value=5)
    
    run_name = st.text_input("🔹 Nhập tên Run:", "Default_Run")  # Tên run cho MLflow
    st.session_state["run_name"] = run_name if run_name else "default_run"
    
    if st.button("Huấn luyện mô hình"):
        with mlflow.start_run(run_name=f"Train_{st.session_state['run_name']}"):
            
            mlflow.log_param("test_size", st.session_state.test_size)
            mlflow.log_param("val_size", st.session_state.val_size)
            mlflow.log_param("train_size", st.session_state.train_size)
            mlflow.log_param("num_samples", st.session_state.total_samples)
            
            os.makedirs("mlflow_artifacts", exist_ok=True)
            np.save("mlflow_artifacts/X_train.npy", X_train)
            np.save("mlflow_artifacts/X_test.npy", X_test)
            np.save("mlflow_artifacts/y_train.npy", y_train)
            np.save("mlflow_artifacts/y_test.npy", y_test)
            mlflow.log_artifacts("mlflow_artifacts")
            
            
            
            # 🏆 **Huấn luyện với Cross Validation**
            st.write("⏳ Đang chạy Cross-Validation...")
            cv_scores = cross_val_score(model, X_train, y_train, cv=n_folds)
            mean_cv_score = cv_scores.mean()
            std_cv_score = cv_scores.std()
            
            st.success(f"📊 **Cross-Validation Accuracy**: {mean_cv_score:.4f}")

            # Huấn luyện mô hình trên tập train chính
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            st.success(f"✅ Độ chính xác trên test set: {acc:.4f}")

            # 📝 Ghi log vào MLflow
            mlflow.log_param("model", model_choice)
            if model_choice == "Decision Tree":
                mlflow.log_param("max_depth", max_depth)
            elif model_choice == "SVM":
                mlflow.log_param("C", C)
                mlflow.log_param("kernel", kernel)

            mlflow.log_metric("test_accuracy", acc)
            mlflow.log_metric("cv_accuracy_mean", mean_cv_score)
            mlflow.log_metric("cv_accuracy_std", std_cv_score)
            mlflow.sklearn.log_model(model, model_choice.lower())

        # Lưu mô hình vào session_state
        if "models" not in st.session_state:
            st.session_state["models"] = []

        model_name = model_choice.lower().replace(" ", "_")
        if model_choice == "SVM":
            model_name += f"_{kernel}"

        existing_model = next((item for item in st.session_state["models"] if item["name"] == model_name), None)

        if existing_model:
            count = 1
            new_model_name = f"{model_name}_{count}"
            while any(item["name"] == new_model_name for item in st.session_state["models"]):
                count += 1
                new_model_name = f"{model_name}_{count}"
            model_name = new_model_name
            st.warning(f"⚠️ Mô hình được lưu với tên: {model_name}")

        st.session_state["models"].append({"name": model_name, "model": model})
        st.write(f"🔹 Mô hình đã được lưu với tên: {model_name}")
        st.write(f"Tổng số mô hình hiện tại: {len(st.session_state['models'])}")

        # Hiển thị danh sách mô hình
        st.write("📋 Danh sách các mô hình đã lưu:")
        model_names = [model["name"] for model in st.session_state["models"]]
        st.write(", ".join(model_names))

        st.success(f"✅ Đã log dữ liệu cho **Train_{st.session_state['run_name']}**!")
        st.markdown(f"🔗 [Truy cập MLflow UI]({st.session_state['mlflow_url']})")


      


 

        

def load_model(path):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        st.error(f"⚠️ Không tìm thấy mô hình tại `{path}`")
        st.stop()

# ✅ Xử lý ảnh từ canvas (chuẩn 28x28 cho MNIST)
def preprocess_canvas_image(canvas_result):
    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data[:, :, 0].astype(np.uint8))
        img = img.resize((28, 28)).convert("L")  # Resize và chuyển thành grayscale
        img = np.array(img, dtype=np.float32) / 255.0  # Chuẩn hóa về [0, 1]
        return img.reshape(1, -1)  # Chuyển thành vector 1D
    return None

import random
# ✅ Chạy dự đoán
def du_doan():
    st.header("✍️ Vẽ số để dự đoán")

    # 🔹 Danh sách mô hình có sẵn
    models = {
        "SVM Linear": "buoi4/svm_mnist_linear.joblib",
        "SVM Poly": "buoi4/svm_mnist_poly.joblib",
        "SVM Sigmoid": "buoi4/svm_mnist_sigmoid.joblib",
        "SVM RBF": "buoi4/svm_mnist_rbf.joblib",
    }

    # Lấy tên mô hình từ session_state
    model_names = [model["name"] for model in st.session_state.get("models", [])]

    # 📌 Chọn mô hình
    model_option = st.selectbox("🔍 Chọn mô hình:", list(models.keys()) + model_names)

    # Nếu chọn mô hình đã được huấn luyện và lưu trong session_state
    if model_option in model_names:
        model = next(model for model in st.session_state["models"] if model["name"] == model_option)["model"]
    else:
        # Nếu chọn mô hình có sẵn (các mô hình đã được huấn luyện và lưu trữ dưới dạng file
        model = load_model(models[model_option])
        st.success(f"✅ Đã tải mô hình: {model_option}")

    # 🆕 Cập nhật key cho canvas khi nhấn "Tải lại"
    if "key_value" not in st.session_state:
        st.session_state.key_value = str(random.randint(0, 1000000))  # Đổi key thành string

    if st.button("🔄 Tải lại nếu không thấy canvas"):
        st.session_state.key_value = str(random.randint(0, 1000000))  # Đổi key thành string
        #st.rerun()  # Cập nhật lại giao diện để vùng vẽ được làm mới
    
    # ✍️ Vẽ số
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=10,
        stroke_color="white",
        background_color="black",
        height=150,
        width=150,
        drawing_mode="freedraw",
        key=st.session_state.key_value,  # Đảm bảo key là string
        update_streamlit=True
    )

    if st.button("Dự đoán số"):
        img = preprocess_canvas_image(canvas_result)

        if img is not None:
            st.image(Image.fromarray((img.reshape(28, 28) * 255).astype(np.uint8)), caption="Ảnh sau xử lý", width=100)

            # Dự đoán số
            prediction = model.predict(img)
            confidence_scores = model.decision_function(img)  # Lấy điểm số tin cậy

            # Chuyển đổi điểm số tin cậy thành xác suất tương đối
            confidence_scores = np.exp(confidence_scores) / np.sum(np.exp(confidence_scores)) 

            predicted_number = prediction[0]
            max_confidence = np.max(confidence_scores)

            st.subheader(f"🔢 Dự đoán: {predicted_number}")
            st.write(f"📊 Mức độ tin cậy (ước lượng): {max_confidence:.2%}")

            # Hiển thị bảng confidence scores
            prob_df = pd.DataFrame(confidence_scores.reshape(1, -1), columns=[str(i) for i in range(10)]).T
            prob_df.columns = ["Mức độ tin cậy"]
            st.bar_chart(prob_df) 

        else:
            st.error("⚠️ Hãy vẽ một số trước khi bấm Dự đoán!")



from datetime import datetime   
def show_experiment_selector():
    st.title("📊 MLflow Experiments - DAGsHub")

    # Kết nối với DAGsHub MLflow Tracking
    
    # Lấy danh sách tất cả experiments
    experiment_name = "Classification"
    
    # Tìm experiment theo tên
    experiments = mlflow.search_experiments()
    selected_experiment = next((exp for exp in experiments if exp.name == experiment_name), None)

    if not selected_experiment:
        st.error(f"❌ Experiment '{experiment_name}' không tồn tại!")
        return

    st.subheader(f"📌 Experiment: {experiment_name}")
    st.write(f"**Experiment ID:** {selected_experiment.experiment_id}")
    st.write(f"**Trạng thái:** {'Active' if selected_experiment.lifecycle_stage == 'active' else 'Deleted'}")
    st.write(f"**Vị trí lưu trữ:** {selected_experiment.artifact_location}")

    # Lấy danh sách runs trong experiment
    runs = mlflow.search_runs(experiment_ids=[selected_experiment.experiment_id])

    if runs.empty:
        st.warning("⚠ Không có runs nào trong experiment này.")
        return

    st.write("### 🏃‍♂️ Các Runs gần đây:")

    # Lấy danh sách run_name từ params
    run_info = []
    for _, run in runs.iterrows():
        run_id = run["run_id"]
        run_tags = mlflow.get_run(run_id).data.tags
        run_name = run_tags.get("mlflow.runName", f"Run {run_id[:8]}")  # Lấy từ tags
        run_info.append((run_name, run_id))

    # Tạo dictionary để map run_name -> run_id
    run_name_to_id = dict(run_info)
    run_names = list(run_name_to_id.keys())

    # Chọn run theo run_name
    selected_run_name = st.selectbox("🔍 Chọn một run:", run_names)
    selected_run_id = run_name_to_id[selected_run_name]

    # Hiển thị thông tin chi tiết của run được chọn
    selected_run = mlflow.get_run(selected_run_id)

    if selected_run:
        st.subheader(f"📌 Thông tin Run: {selected_run_name}")
        st.write(f"**Run ID:** {selected_run_id}")
        st.write(f"**Trạng thái:** {selected_run.info.status}")
        start_time_ms = selected_run.info.start_time  # Thời gian lưu dưới dạng millisecondT

# Chuyển sang định dạng ngày giờ dễ đọc
        if start_time_ms:
            start_time = datetime.fromtimestamp(start_time_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")
        else:
            start_time = "Không có thông tin"

        st.write(f"**Thời gian chạy:** {start_time}")

        # Hiển thị thông số đã log
        params = selected_run.data.params
        metrics = selected_run.data.metrics

        if params:
            st.write("### ⚙️ Parameters:")
            st.json(params)

        if metrics:
            st.write("### 📊 Metrics:")
            st.json(metrics)

        # Kiểm tra và hiển thị dataset artifact
        dataset_path = f"{selected_experiment.artifact_location}/{selected_run_id}/artifacts/dataset.csv"
        st.write("### 📂 Dataset:")
        st.write(f"📥 [Tải dataset]({dataset_path})")
    else:
        st.warning("⚠ Không tìm thấy thông tin cho run này.")
           
            
            
            
def Classification():
  
    if "mlflow_initialized" not in st.session_state:   
        DAGSHUB_MLFLOW_URI = "https://dagshub.com/Phuocbinh2003/Hoc_may_python.mlflow"
        st.session_state['mlflow_url']=DAGSHUB_MLFLOW_URI
        mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)

        os.environ["MLFLOW_TRACKING_USERNAME"] = "Phuocbinh2003"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "c1495823c8f9156923b06f15899e989db7e62052"
        st.session_state.mlflow_initialized = True
        mlflow.set_experiment("Classification")   
        
    st.title("🖊️ MNIST Classification App")
    
    #st.session_state.clear()
    ### **Phần 1: Hiển thị dữ liệu MNIST**
    
    ### **Phần 2: Trình bày lý thuyết về Decision Tree & SVM*
    
    # 1️⃣ Phần giới thiệu
    
    # === Sidebar để chọn trang ==
    # === Tạo Tabs ===
    tab1, tab2, tab3, tab4,tab5 ,tab6= st.tabs(["📘 Lý thuyết Decision Tree", "📘 Lý thuyết SVM", "📘 Data" ,"⚙️ Huấn luyện", "🔢 Dự đoán","🔥Mlflow"])
    
    with tab1:
        
        ly_thuyet_Decision_tree()

    with tab2:
        ly_thuyet_SVM()
    
    with tab3:
        data()
        
    with tab4:
       # plot_tree_metrics()
        
        
        
        split_data()
        train()
        
    
    with tab5:
        
        du_doan()   
    with tab6:
        
        show_experiment_selector()  




            
if __name__ == "__main__":
    Classification()