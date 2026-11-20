# Báo cáo CRISP-DM: House Prices - Advanced Regression Techniques

## 1. Business Understanding (Hiểu bài toán)

Mục tiêu là ước lượng giá bán cuối cùng (`SalePrice`) của mỗi căn nhà tại
Ames, Iowa từ các đặc trưng về khu đất, chất lượng xây dựng, diện tích, gara,
tầng hầm, tiện ích và thời gian. Đây là bài toán **hồi quy có giám sát**:
đầu vào là 79 biến mô tả, đầu ra là một giá liên tục. Kết quả được dùng để
hỗ trợ định giá và xếp hạng các căn nhà.

Cuộc thi Kaggle đánh giá bằng **Root Mean Squared Logarithmic Error
(RMSLE)**, tương đương RMSE giữa `log(1 + giá thật)` và `log(1 + giá dự đoán)`.
Vì vậy mô hình học `log1p(SalePrice)` và đảo biến bằng `expm1` khi tạo
submission.

## 2. Data Understanding (Hiểu dữ liệu)

Nguồn dữ liệu là bộ Ames Housing do De Cock mô tả trong bài báo *Ames,
Iowa: Alternative to the Boston Housing Data as an End of Sample
Benchmark for Regression* (Journal of Statistics Education, 2011). Tập
`train.csv` có 1.460 dòng và 81 cột (gồm nhãn), còn `test.csv` có 1.459 dòng
và 80 cột. `SalePrice` lệch phải mạnh (skewness khoảng 1,88), vì vậy biến đổi
log là phù hợp. Dữ liệu có nhiều giá trị thiếu, trong đó thiếu đôi khi mang
ý nghĩa “không có hạng mục” (ví dụ không có gara hoặc hồ bơi), nhưng để giữ
pipeline nhất quán ta điền mode cho biến phân loại và median cho biến số.

Không sử dụng `Id` làm biến dự báo vì đây là mã định danh. Tập kiểm tra không
có `SalePrice`, do đó một phần 20% của train được giữ lại để kiểm tra trong
quá trình phát triển.

## 3. Data Preparation (Chuẩn bị dữ liệu)

1. Đọc train/test và tách `SalePrice`.
2. Biến đổi nhãn bằng `log1p`.
3. Xóa `Id`; ghép train và test **chỉ ở ma trận đặc trưng** để hai tập có
   cùng tập cột one-hot.
4. Điền median cho biến số và mode cho biến phân loại.
5. One-hot encoding biến phân loại.
6. Chuẩn hóa từng cột theo trung bình và độ lệch chuẩn của train.
7. Chia train thành 80% huấn luyện và 20% validation với `random_state=42`.

Các bước này nằm trong `prepare_features` của `train_mlp.py`. Việc không dùng
nhãn của test và chỉ tính thống kê chuẩn hóa từ train giúp tránh rò rỉ nhãn.

## 4. Modeling (Mô hình hóa)

Mô hình là MLP viết bằng PyTorch:

- Dense `n_features -> 256 -> 128 -> 64 -> 1`.
- ReLU, BatchNorm và Dropout (0,20 rồi 0,15).
- Hàm mất mát MSE trên log-price.
- AdamW, learning rate ban đầu `1e-3`, weight decay `1e-4`.
- Giảm learning rate khi validation không cải thiện và early stopping sau
  50 epoch không cải thiện.
- Clip gradient ở norm 5 để huấn luyện ổn định.

Checkpoint tốt nhất được lưu thành `outputs/best_model.pt`. Dùng
`python train_mlp.py --epochs 500 --patience 50` để chạy toàn bộ thí nghiệm.

## 5. Evaluation (Đánh giá)

`valid_rmse_log` là metric chính, cùng quy ước với RMSLE của Kaggle; MAE trên
log-price được ghi thêm để tham khảo. Kết quả thực tế được lưu trong
`outputs/metrics.json` sau khi chạy. Vì validation chỉ là một lần chia ngẫu
nhiên, điểm Kaggle có thể khác; cần nộp `outputs/submission.csv` trên trang
competition để nhận điểm kiểm tra chính thức.

Các kiểm tra đầu ra cần thực hiện:

- `submission.csv` có đúng hai cột `Id,SalePrice`.
- Có 1.459 dòng dự đoán và không có giá trị thiếu/âm.
- `history.csv` có loss/metric theo epoch.
- `metrics.json` ghi rõ seed, thiết bị, số đặc trưng và epoch tốt nhất.

## 6. Deployment (Triển khai)

File `outputs/submission.csv` được tải lên mục **Submit Predictions** của
[House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).
Để tái lập, chạy lệnh trong README từ thư mục bài thực hành. Có thể cải thiện
tiếp bằng cross-validation, xử lý ngữ nghĩa “không có hạng mục” riêng, log
các biến lệch phải, ensemble với mô hình cây/gradient boosting và tinh chỉnh
siêu tham số.

## Tài liệu tham khảo

1. De Cock, D. (2011). *Ames, Iowa: Alternative to the Boston Housing Data
   as an End of Sample Benchmark for Regression*. Journal of Statistics
   Education, 19(3). `decock.pdf` được cung cấp cùng dữ liệu.
2. Kaggle, *House Prices - Advanced Regression Techniques* và
   `data_description.txt` đi kèm bộ dữ liệu.
