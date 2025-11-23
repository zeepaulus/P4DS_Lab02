# BANK CUSTOMER CHURN PREDICTION

Dự đoán Khả năng Rời bỏ Ngân hàng (Customer Attrition Analysis) bằng Logistic Regression và KNN hoàn toàn bằng Numpy.

---

# Mục lục
1. [Giới thiệu](#giới-thiệu)
2. [Dataset](#dataset)
3. [Method](#method)
4. [Installation & Setup](#installation--setup)
5. [Usage](#usage)
6. [Results](#results)
7. [Project Structure](#project-structure)
8. [Challenges & Solutions](#challenges--solutions)
9. [Future Improvements](#future-improvements)
10. [Contributors](#contributors)
11. [License](#license)

---

# 1. Giới thiệu

### Mô tả bài toán
Dựa trên thông tin cá nhân, hành vi giao dịch, và lịch sử sử dụng tín dụng, project này xây dựng một mô hình học máy để dự đoán khả năng một khách hàng sẽ **ngừng sử dụng dịch vụ của ngân hàng** (`Attrited Customer`).

### Động lực và ứng dụng thực tế
Việc dự đoán churn sớm giúp ngân hàng và phòng Marketing:
- Hiểu rõ các yếu tố hành vi (như sự sụt giảm giao dịch gần đây) ảnh hưởng đến quyết định rời bỏ.
- Xác định các nhóm khách hàng có nguy cơ nghỉ việc cao (Risk Segments).
- Chủ động đưa ra các chính sách đãi ngộ hoặc cải tiến dịch vụ để giữ chân khách hàng.

### Mục tiêu cụ thể
1.  **Phân tích dữ liệu khám phá (EDA)** để tìm ra các insight quan trọng về hành vi tài chính.
2.  **Xây dựng một pipeline tiền xử lý dữ liệu** hoàn chỉnh chỉ bằng NumPy.
3.  **Cài đặt lại từ đầu (from scratch) mô hình Logistic Regression** và **KNN** cùng các hàm đánh giá bằng NumPy.
4.  **Huấn luyện và tinh chỉnh** mô hình để đạt được hiệu suất dự đoán tốt nhất.

---

# 2. Dataset

## Nguồn dữ liệu
Dữ liệu được sử dụng là bộ dữ liệu **Credit Card Customers Churn Prediction** (BankChurners.csv).

### Mô tả các features
- `Attrition_Flag`: Trạng thái Khách hàng (Target)
- `Customer_Age`: Tuổi của khách hàng
- `Gender`: Giới tính của khách hàng
- `Dependent_count`:	Số người phụ thuộc
- `Education_Level`:	Trình độ học vấn
- `Marital_Status`: Tình trạng hôn nhân
- `Income_Category`:	Nhóm thu nhập
- `Card_Category`: Loại thẻ tín dụng
- `Months_on_book`: Số tháng gắn bó với ngân hàng
- `Total_Relationship_Count`:	Tổng số sản phẩm đang sử dụng
- `Months_Inactive_12_mon`: Số tháng không hoạt động trong vòng 12 tháng gần nhất
- `Contacts_Count_12_mon`: Số lần liên hệ với ngân hàng trong vòng 12 tháng gần nhất
- `Credit_Limit`: Hạn mức tín dụng
- `Total_Revolving_Bal`	Số dư quay vòng
- `Avg_Open_To_Buy`: Hạn mức khả dụng trung bình
- `Total_Amt_Chng_Q4_Q1`: Tỷ lệ giá trị giao dịch giữa Quý 4 và Quý 1
- `Total_Trans_Amt`: Tổng giá trị giao dịch trong 12 tháng
- `Total_Trans_Ct`: Tổng số giao dịch trong 12 tháng
- `Total_Ct_Chng_Q4_Q1`: Tỷ lệ số lần giao dịch giữa Quý 4 và Quý 1
- `Avg_Utilization_Ratio`: Tỷ lệ sử dụng hạn mức tín dụng trung bình

## Kích thước và đặc điểm
- **Tổng**: 10,127 samples
- **Missing, Duplicate values**: Không có
- **Số features**: 22 features (ban đầu), 16 features (sau khi xử lý dữ liệu)
- **Data types**: chứa cả dữ liệu categorical (6 features) và numerical (10 features)
---

# 3. Method

### Quy trình xử lý dữ liệu
1. **Data Loading**: load dữ liệu bằng Numpy
2. **Data Cleaning:** Loại bỏ các cột thừa (`CLIENTNUM`, `Naive_Bayes_...`)
3. **Xác định Target**: Tách `Attrition_Flag` khỏi data
4. **Mã hóa Encoding:** Sử dụng **One-hot Encoding** cho các biến phân loại (Categorical)
5. **Chuẩn hóa Standardization:** Áp dụng chuẩn hóa **Z-score** cho các biến số (`Credit_Limit`, `Total_Trans_Amt`, etc.)
6. **Feature Selection:** Loại bỏ các biến đa cộng tuyến (ví dụ: `Avg_Open_To_Buy`, etc.)
7. **Chia dữ liệu**: 80% dùng để train, 20% dùng để test

## Thuật toán sử dụng

Project sử dụng hai thuật toán được cài đặt lại từ đầu (from scratch) bằng NumPy:

### Logistic Regression

Là mô hình phân loại tuyến tính chuẩn mực cho bài toán nhị phân. Model sử dụng hàm Sigmoid để chuyển đổi đầu ra tuyến tính thành xác suất, sau đó tối ưu bằng Gradient Descent để giảm thiểu hàm mất mát Binary Cross-Entropy.

- **Binary Cross-Entropy (Cost Function) với L2 Regularization:**
```math
J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)}\log(a^{(i)}) + (1-y^{(i)})\log(1-a^{(i)})] + \frac{\alpha}{2m} \sum_{j=1}^{n} w_j^2
```

- **Hàm Sigmoid:**
```math
\sigma(z) = \frac{1}{1 + e^{-z}}
```

- **Gradient Descent Update:**
```math
w_j := w_j - \eta \frac{\partial J}{\partial w_j}
```
```math
b_j := b_j - \eta \frac{\partial J}{\partial b_j}
```

### K-Nearest Neighbors (KNN)

Là thuật toán Học lười (Lazy Learning) dựa trên khoảng cách. Nó phân loại một điểm dữ liệu mới bằng cách tìm kiếm $K$ điểm gần nhất và áp dụng nguyên tắc đa số phiếu bầu.

- **Distance Metric**: dùng khoảng cách Euclidean để đo độ tương đồng
```math
d(\mathbf{x}_i, \mathbf{x}_j) = \sqrt{\sum_{k=1}^{n} (x_{ik} - x_{jk})^2}
```

- **Classification Rule**: dựa nhãn phổ biến nhất trong $K$ điểm lân cận
```math
\hat{y} = \arg\max_{v \in \{0, 1\}} \sum_{i \in N_k(\mathbf{x})} I(y_i = v)
```
Trong đó $N_k(\mathbf{x})$ là tập hợp $K$ láng giềng gần nhất, và $I$ là hàm chỉ thị

## Implement với NumPy

**1. Kỹ thuật vector hóa và ma trận**  
- **Vectorization & Matrix Operations:** Sử dụng các phép toán ma trận (`np.dot`) thay thế hoàn toàn cho các vòng lặp Python và giúp tăng tốc đáng kể.  
- **KNN Distance:** Khoảng cách Euclidean giữa một điểm và toàn bộ tập train được tính bằng các phép toán mảng (broadcasting, `np.sum(axis=1)`, `np.sqrt`), cho phép xử lý hàng loạt điểm một cách hiệu quả.

**2. Xử lý dữ liệu**  
- **Standardization:** Chuẩn hóa dữ liệu số theo Z‑score \(\frac{x - \mu}{\sigma}\) với `np.mean`, `np.std` theo từng cột, sau đó áp dụng broadcasting để đưa tất cả feature về cùng thang đo.  
- **Numerical Stability:** Dùng `np.clip()` để giới hạn đầu vào của sigmoid và giá trị xác suất dự đoán trước khi tính `log`, tránh overflow của `exp()` và lỗi \(\log(0)\) dẫn tới \(-\infty\).

**3. Phân loại và đánh giá**  
- **Metrics vectorized:** Các chỉ số như Accuracy, F1‑Score, Confusion Matrix được tính bằng boolean masking trên mảng (so sánh `y_true`, `y_pred`) kết hợp với `np.sum`, giúp đánh giá mô hình nhanh và gọn.  
- **Regularization:** L2 regularization được cộng trực tiếp vào hàm mất mát và gradient update, kiểm soát độ lớn của trọng số và giảm nguy cơ overfitting.  

# 4. Installation & Setup

1. Clone repository:
```bash
git clone <repository-url>
cd P4DS_Lab02
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
python -m notebook
```

# 5. Usage

Chạy các notebooks theo thứ tự sau:

## `01_data_exploration.ipynb`
Run các cell theo thứ tự từ trên xuống. File này thực hiện:
- Cho cái nhìn tổng quan về dữ liệu đang làm.
- Trực quan hóa dữ liệu thông qua đa dạng các biểu đồ.
- Tìm hiểu ý nghĩa của các features, cách nó ảnh hưởng đến Attrited/Existing Customer.

## `02_preprocessing.ipynb`
Run các cell theo thứ tự từ trên xuống. File này thực hiện:
- Loại bỏ các feature dư thừa, không mang ý nghĩa.
- Xử lý dữ liệu categorical bằng One-hot Encoding.
- Chuẩn hóa dữ liệu số.
- Vẽ biểu đồ nhiệt để loại bỏ các biến đa cộng tuyến.
- Phân chia dữ liệu thành tập train/test.

## `03_modeling.ipynb`
Run các cell theo thứ tự từ trên xuống. File này thực hiện:
- Huấn luyện mô hình Logistic Regression và K-Nearest Neighbors.
- Tính các thông số như: accuracy, precision, recall, F1, AUC.
- Vẽ confusion matrix, ROC.
- So sánh 2 mô hình.

# 6. Results
## Kết quả đạt được:
| Metric | Logistic Regression | KNN (K=13) |
| :--- | :--- | :--- |
| **Accuracy** | 0.8805 | 0.9022 |
| **Precision** | 0.6261 | 0.8728 |
| **Recall** | 0.6453 | 0.4618 |
| **F1-Score** | 0.6355 | 0.6040 |

##  Trực quan hoá kết quả:
**Confusion matrix**
<img width="1920" height="833" alt="confusion_matrix" src="https://github.com/user-attachments/assets/84862f34-d597-48cf-beb7-aa330c8e05d2" />
**So sánh performance**
<img width="1590" height="593" alt="output" src="https://github.com/user-attachments/assets/0f254590-c8f9-474a-a32e-0c0ead4d4028" />

# 7. Project Structure
```
project/
├── README.md                
├── requirements.txt          
├── LICENSE                   
├── data/
│   ├── raw/                 
│   │   └── BankChurners.csv
│   └── processed/           
├── notebooks/
│   ├── 01_data_exploration.ipynb    
│   ├── 02_preprocessing.ipynb       
│   └── 03_modeling.ipynb            
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── visualization.py
│   └── models.py 
├── .gitignore
└── README.md   <- bạn đang ở đây
```

# 8. Challenges & Solutions
**Challenges**: Ràng buộc chỉ sử dụng NumPy cho toàn bộ pipeline.

**Solutions**:
- Vectorization: Biểu diễn toàn bộ tính toán bằng các phép toán trên array thay cho vòng lặp.
- Broadcasting: Tận dụng broadcasting khi chuẩn hóa dữ liệu, cập nhật tham số và áp dụng ngưỡng để giảm code lặp và tăng hiệu năng.
- Advanced indexing: Sử dụng boolean mask và fancy indexing để lọc dữ liệu, chia train–test, tính TP/FP/FN/TN mà không cần duyệt từng phần tử.

# 9. Future Improvements
- **Cân bằng dữ liệu & mở rộng feature**: Sử dụng các kỹ thuật oversampling (như SMOTE) cho lớp thiểu số, đồng thời tạo thêm đặc trưng.
- **Mô hình nâng cao**: Bổ sung các thuật toán khác như Decision Tree, Random Forest, mạng nơ‑ron đơn giản và các mô hình ensemble (bagging, voting).
- **Giải thích & chọn lọc feature**: Áp dụng các phương pháp xAI (như SHAP) để hiểu đóng góp của từng feature và đánh giá mức độ quan trọng của chúng.
- **Triển khai ứng dụng**: Đóng gói mô hình vào API (Flask/FastAPI) phục vụ dự đoán thời gian thực trên dữ liệu thực tế.

# 10. Contributors
**Sinh viên**
- Họ tên: Bùi Duy Bảo
- MSSV: 23122021
**Contact**
- Email: 23122021@student.hcmus.edu.vn

# 11. License
MIT License - See `LICENSE` file for details
