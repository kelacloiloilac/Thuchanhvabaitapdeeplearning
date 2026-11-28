# BÁO CÁO BÀI TẬP LỚN

## Nhận diện cảm xúc khuôn mặt trên bộ dữ liệu FER2013 bằng mạng nơ-ron tích chập

**Môn học:** Học sâu / Deep Learning
**Đề tài:** Facial Emotion Recognition – FER2013
**Sinh viên thực hiện:** Nguyễn Trần Phi Anh, Nguyễn Cảnh Thắng
**Giảng viên hướng dẫn:** Đỗ Như Tài
**Thời gian:** 28/11/2026

---

## Tóm tắt

Bài tập lớn nghiên cứu bài toán phân loại bảy trạng thái cảm xúc khuôn mặt (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) trên bộ dữ liệu FER2013 gồm 35.887 ảnh xám kích thước 48×48. Chúng tôi cài đặt một mạng nơ-ron tích chập (CNN) bằng PyTorch với ba cấu hình kiến trúc (`small`, `medium`, `small_nobn`) và tiến hành sáu thí nghiệm E1–E6 nhằm khảo sát ảnh hưởng của tăng cường dữ liệu, chuẩn hóa theo lô (BatchNorm), bộ tối ưu (Adam/SGD), bộ điều chỉnh tốc độ học (OneCycle) và làm mịn nhãn (label smoothing).

Ở cấu hình cơ sở, mô hình tốt nhất `E6_medium_aug_onecycle` đạt độ chính xác **60,13%** trên tập PrivateTest (F1 macro 0,484); lớp Happy đạt F1 cao nhất 0,844, còn lớp Disgust gần như không được nhận diện (F1 = 0) do mất cân bằng dữ liệu nghiêm trọng. Phiên bản cải tiến bổ sung **class weights dạng sqrt**, **random erasing**, tăng cường xoay/dịch/zoom mạnh hơn và **huấn luyện dài hơn** (20–30 epoch). Kết quả: mô hình tốt nhất đạt **64,84%** PrivateTest (+4,71 điểm phần trăm), F1 macro tăng lên 0,600; đặc biệt Disgust từ F1 = 0 lên **0,496** và Fear từ 0,256 lên 0,344, cho thấy hiệu quả của cân bằng lớp.  phân tích ba mức class weights cho thấy `sqrt` là điểm cân bằng tốt nhất: `none` bỏ rơi Disgust (F1 = 0,102), còn `inverse` kéo recall Disgust lên 0,836 nhưng precision chỉ 0,198 và làm accuracy giảm còn 61,10%.

---

## Mục lục

1. [Chương 1 – Giới thiệu](#chương-1--giới-thiệu)
2. [Chương 2 – Cơ sở lý thuyết](#chương-2--cơ-sở-lý-thuyết)
3. [Chương 3 – Dữ liệu và tiền xử lý](#chương-3--dữ-liệu-và-tiền-xử-lý)
4. [Chương 4 – Phương pháp và thiết kế thực nghiệm](#chương-4--phương-pháp-và-thiết-kế-thực-nghiệm)
5. [Chương 5 – Kết quả thực nghiệm và đánh giá](#chương-5--kết-quả-thực-nghiệm-và-đánh-giá)
6. [Chương 6 – Kết luận và hướng phát triển](#chương-6--kết-luận-và-hướng-phát-triển)
7. [Tài liệu tham khảo](#tài-liệu-tham-khảo)
8. [Phụ lục](#phụ-lục)

---

## Danh mục từ viết tắt

| Từ viết tắt | Ý nghĩa                                                       |
| -------------- | --------------------------------------------------------------- |
| CNN            | Convolutional Neural Network – mạng nơ-ron tích chập       |
| BN             | Batch Normalization – chuẩn hóa theo lô                     |
| FER            | Facial Emotion Recognition – nhận diện cảm xúc khuôn mặt |
| FC             | Fully Connected – lớp kết nối đầy đủ                    |
| LR             | Learning Rate – tốc độ học                                 |
| WD             | Weight Decay – suy giảm trọng số                            |
| LS             | Label Smoothing – làm mịn nhãn                              |
| HCI            | Human–Computer Interaction – tương tác người–máy       |
| P/R/F1         | Precision / Recall / F1-score                                   |

---

# Chương 1 – Giới thiệu

## 1.1. Đặt vấn đề

Nhận diện cảm xúc khuôn mặt là một bài toán quan trọng của thị giác máy tính, có nhiều ứng dụng trong tương tác người–máy, giám sát trạng thái học tập, chăm sóc sức khỏe tinh thần và phân tích hành vi người dùng. So với các bài toán phân loại ảnh thông thường, bài toán này khó vì biểu cảm con người rất đa dạng, sự khác biệt giữa các lớp có thể nhỏ, ảnh có độ phân giải thấp và dữ liệu thường mất cân bằng.

## 1.2. Mục tiêu của bài tập lớn

- Tìm hiểu và cài đặt mạng CNN cho bài toán phân loại cảm xúc trên FER2013.
- Xây dựng quy trình huấn luyện – đánh giá – phân tích tái lập được bằng PyTorch.
- Khảo sát ảnh hưởng của các yếu tố: tăng cường dữ liệu, BatchNorm, bộ tối ưu, lịch tốc độ học, label smoothing và cân bằng lớp.
- Phân tích lỗi và trực quan hóa đặc trưng để hiểu hành vi của mô hình.

## 1.3. Phạm vi và đối tượng

- Bộ dữ liệu FER2013: 35.887 ảnh xám 48×48, 7 lớp cảm xúc.
- Phân chia chuẩn: Training (28.709), PublicTest (3.589), PrivateTest (3.589).
- Mô hình: CNN huấn luyện từ đầu, không dùng pre-trained.
- Đánh giá chính trên tập PrivateTest (đóng vai trò tập kiểm thử).

## 1.4. Bố cục báo cáo

Chương 2 trình bày cơ sở lý thuyết; Chương 3 mô tả dữ liệu và tiền xử lý; Chương 4 trình bày phương pháp, kiến trúc và thiết kế thực nghiệm; Chương 5 báo cáo kết quả và phân tích; Chương 6 kết luận và hướng phát triển.

## 1.5. Câu hỏi nghiên cứu

Báo cáo tập trung trả lời ba câu hỏi:

- **RQ1:** Class weights dạng `sqrt` có cải thiện F1/recall của các lớp thiểu số (Disgust, Fear) mà không làm giảm đáng kể độ chính xác chung không?
- **RQ2:** Sau khi cải tiến, những cặp lớp nào vẫn bị nhầm lẫn nhiều nhất, và điều đó phản ánh giới hạn gì của CNN huấn luyện từ đầu trên ảnh 48×48?

## 1.6. Đóng góp chính

1. Thiết kế và chạy hệ thống E1–E6 để tách ảnh hưởng của augmentation, BatchNorm, optimizer, scheduler và label smoothing.
2. Đề xuất và kiểm chứng cập nhật `long` + `sqrt` + random erasing, so sánh trực tiếp với mô hình gốc: **+4,71 điểm phần trăm** accuracy và **+0,116** macro F1; kèm  phân tích `none`/`sqrt`/`inverse` để kiểm tra trade-off recall–precision của lớp thiểu số.
3. Phân tích lỗi và trực quan hóa đặc trưng để giải thích hành vi mô hình.

## 1.7. Phân công nhiệm vụ

Bài tập lớn được thực hiện bởi hai thành viên với mức đóng góp **50% – 50%**. Bảng dưới đây tóm tắt hạng mục chính và nhiệm vụ chi tiết của từng thành viên.

| Thành viên | Hạng mục chính | Nhiệm vụ chi tiết | Tỷ lệ đóng góp |
| ------------ | --------------- | ---------------------------------------------------------- | ---------- |
| Nguyễn Trần Phi Anh | Mô hình, tối ưu hóa và báo cáo lý thuyết | • **Code:** xây dựng kiến trúc mạng CNN (`small`, `medium`, `small_nobn` trong `models.py`); cài đặt vòng lặp huấn luyện, hàm loss Cross-Entropy có Label Smoothing, bộ tối ưu Adam/SGD và scheduler OneCycleLR (`engine.py`).<br>• **Thực nghiệm:** huấn luyện và đánh giá đợt thực nghiệm cơ sở E1–E6 (10–12 epoch); kiểm chứng ảnh hưởng của BatchNorm và scheduler.<br>• **Báo cáo:** soạn thảo Chương 1 (Giới thiệu), Chương 2 (Cơ sở lý thuyết toán học), Chương 4 (Phương pháp và thiết kế thực nghiệm), Chương 6 (Kết luận và hướng phát triển).<br>• **Nhật ký Excel:** thiết lập bảng tính toán lý thuyết số lượng tham số từng tầng của 3 kiến trúc mạng và số bước lặp (Steps/Epoch).<br>• **Slide:** chuẩn bị nội dung slide nửa đầu: giới thiệu đề tài, mục tiêu, cơ sở lý thuyết mạng CNN, thiết kế kiến trúc và bộ tối ưu hóa. | 50% (hoàn thành 100%) |
| Nguyễn Cảnh Thắng | Dữ liệu, cân bằng lớp, trực quan hóa và slide | • **Code:** tiền xử lý dữ liệu từ CSV sang NPZ (`prepare_data.py`, `data.py`); xây dựng module Data Augmentation (Affine Grid) và Random Erasing (`augment.py`); cài đặt Class Weights (`sqrt`, `inverse`).<br>• **Thực nghiệm:** triển khai đợt thực nghiệm mở rộng 20–30 epoch trên Colab GPU (`outputs_long_*`), tối ưu hóa mô hình đạt kỷ lục 64,84%.<br>• **Trực quan hóa:** lập trình xuất biểu đồ hàm mất mát/accuracy, ma trận nhầm lẫn (Confusion Matrix), bộ lọc Conv cuối (Activation Maximization, Feature Maps) và ảnh dự đoán đúng/sai (`viz.py`, `plots.py`, `analyze.py`).<br>• **Báo cáo:** soạn thảo Chương 3 (Dữ liệu và tiền xử lý), Chương 5 (Kết quả thực nghiệm, phân tích về Class Weights, phân tích lỗi và trực quan hóa), Phụ lục mã nguồn.<br>• **Nhật ký Excel:** hoàn thiện bảng phân bố dữ liệu 7 lớp, công thức tính Class Weights sống và xuất lịch sử chi tiết từng epoch.<br>• **Slide:** thiết kế toàn bộ slide (bố cục, đồ họa, biểu đồ trực quan); chuẩn bị nội dung slide nửa sau: tiền xử lý dữ liệu, kết quả thực nghiệm đối đầu, trực quan hóa Filter/Feature Maps và demo nhận diện. | 50% (hoàn thành 100%) |

---

# Chương 2 – Cơ sở lý thuyết

Chương này trình bày cơ sở lý thuyết cho bài toán phân loại cảm xúc khuôn mặt trên
FER2013: phát biểu bài toán phân loại ảnh, cấu trúc mạng CNN, các kỹ thuật huấn
luyện, xử lý mất cân bằng lớp, tăng cường dữ liệu và hệ thống độ đo đánh giá.

## 2.1. Bài toán phân loại ảnh và CNN

### 2.1.1. Phát biểu bài toán

Cho ảnh đầu vào $x \in \mathbb{R}^{48 \times 48 \times 1}$, tương ứng ảnh xám kích
thước $48\times48$. Mô hình cần dự đoán nhãn

$$
y \in \{0,1,\dots,6\},
$$

tương ứng bảy cảm xúc: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.

Với mỗi ảnh $x_i$, mô hình tham số $\theta$ sinh ra vector logits

$$
z_i = f_\theta(x_i) \in \mathbb{R}^7.
$$

Xác suất dự đoán cho từng lớp được tính bằng softmax:

$$
p_\theta(y=k\mid x_i)
=
\frac{\exp(z_{i,k})}{\sum_{j=0}^{6}\exp(z_{i,j})}.
$$

Nhãn dự đoán là

$$
\hat y_i = \arg\max_{k} p_\theta(y=k\mid x_i).
$$

Mục tiêu huấn luyện là tìm $\theta$ sao cho rủi ro thực nghiệm trên tập huấn luyện
là nhỏ nhất:

$$
\min_\theta
\frac{1}{N}\sum_{i=1}^{N}
\mathcal{L}\big(f_\theta(x_i), y_i\big),
$$

trong đó $N$ là số ảnh huấn luyện và $\mathcal{L}$ là hàm mất mát, thường là
cross-entropy.

### 2.1.2. Vì sao dùng CNN cho bài toán này?

CNN phù hợp với ảnh vì ba tính chất quan trọng:

1. **Kết nối cục bộ:** mỗi nơ-ron ở tầng tích chập chỉ nhìn một vùng nhỏ của ảnh, phù hợp với đặc trưng cục bộ như cạnh, mắt, miệng, lông mày.
2. **Chia sẻ trọng số:** cùng một bộ lọc được áp dụng trên toàn ảnh, giúp giảm mạnh số tham số so với mạng fully-connected và tăng khả năng tổng quát hóa.
3. **Học đặc trưng phân cấp:** các tầng đầu học cạnh/biên; các tầng giữa học bộ phận khuôn mặt; các tầng sau tổ hợp thành biểu cảm. Với FER, thông tin hình học và quan hệ giữa mắt–miệng–lông mày rất quan trọng, nên CNN có lợi thế so với đặc trưng thủ công.

Tích chập cho ta tính **equivariance tịnh tiến**: nếu vật thể dịch đi một ít, đặc trưng tích chập cũng dịch theo. Pooling và các tầng sâu hơn tạo ra **invariance gần đúng**, giúp mô hình bền vững hơn với thay đổi vị trí nhỏ.

### 2.1.3. Kích thước không gian và vùng tiếp nhận

Với một tầng tích chập có kernel $K$, padding $P$, stride $S$, kích thước đầu ra được tính bởi

$$
H_{\text{out}}
=
\left\lfloor
\frac{H_{\text{in}} + 2P - K}{S}
\right\rfloor + 1.
$$

Với Conv $3\times3$, padding 1, stride 1 trên ảnh $48\times48$, kích thước không gian được giữ nguyên:

$$
H_{\text{out}} = 48.
$$

MaxPooling $2\times2$ thường làm giảm một nửa kích thước không gian. Vùng tiếp nhận
$r_l$ của một nơ-ron ở tầng $l$ được mở rộng dần theo độ sâu:

$$
r_l = r_{l-1} + (K_l - 1)\prod_{i<l} S_i,
\qquad r_0 = 1,
$$

trong đó $S_i$ là stride của tầng $i$ (bao gồm cả pooling). Nhờ đó, các tầng sâu có
thể nhìn được vùng khuôn mặt lớn hơn, phù hợp để học biểu cảm tổng thể thay vì
chỉ cạnh cục bộ.

---

## 2.2. Các thành phần của mạng

### 2.2.1. Tầng tích chập Conv2D 3×3, padding 1

Tầng tích chập tính

$$
y_{i,j,o}
=
b_o
+
\sum_{c=1}^{C_{\text{in}}}
\sum_{u=0}^{K-1}
\sum_{v=0}^{K-1}
W_{o,c,u,v}\,
x_{i+u,j+v,c},
$$

trong đó $o$ là chỉ số kênh đầu ra, $c$ là kênh đầu vào, $K=3$. Số tham số của một tầng Conv2D là

$$
K^2 \cdot C_{\text{in}} \cdot C_{\text{out}} + C_{\text{out}}.
$$

Kernel $3\times3$ là lựa chọn cân bằng giữa khả năng biểu diễn và chi phí tính toán. Hai tầng $3\times3$ xếp liên tiếp có vùng tiếp nhận tương đương một tầng $5\times5$ nhưng ít tham số hơn và có thêm phi tuyến ở giữa. Padding 1 giúp giữ kích thước không gian, thuận lợi cho thiết kế mạng sâu.

### 2.2.2. Batch Normalization

Với một mini-batch $\mathcal{B}$, BatchNorm chuẩn hóa đầu ra theo từng kênh:

$$
\mu_{\mathcal B}
=
\frac{1}{|\mathcal B|}
\sum_{i\in\mathcal B} x_i,
\qquad
\sigma_{\mathcal B}^2
=
\frac{1}{|\mathcal B|}
\sum_{i\in\mathcal B}(x_i-\mu_{\mathcal B})^2,
$$

$$
\hat x_i
=
\frac{x_i-\mu_{\mathcal B}}{\sqrt{\sigma_{\mathcal B}^2+\epsilon}},
\qquad
y_i = \gamma \hat x_i + \beta.
$$

Trong huấn luyện, BatchNorm dùng thống kê của batch hiện tại và cập nhật running mean/var. Khi suy luận, mô hình dùng running mean/var thay vì thống kê batch BatchNorm giúp quá trình tối ưu ổn định hơn, cho phép learning rate lớn hơn, giảm độ nhạy với khởi tạo và có tác dụng chuẩn hóa nhẹ. Tuy nhiên, với batch quá nhỏ, thống kê batch nhiễu và có thể làm giảm hiệu năng. Trong mã nguồn, BatchNorm dùng mặc định của PyTorch: $\epsilon=10^{-5}$, momentum $=0{,}1$.

### 2.2.3. ReLU

ReLU được định nghĩa:

$$
\mathrm{ReLU}(x)=\max(0,x),
\qquad
\mathrm{ReLU}'(x)=
\begin{cases}
1, & x>0,\\
0, & x\le 0.
\end{cases}
$$

ReLU tạo phi tuyến, giúp mạng xấp xỉ hàm phức tạp, đồng thời giảm triệt tiêu gradient so với sigmoid/tanh ở miền dương. Nhược điểm là hiện tượng “dying ReLU” khi nhiều nơ-ron luôn ở trạng thái âm. Trong thực tế, ReLU thường được đặt sau Conv2D và BatchNorm.

### 2.2.4. MaxPooling 2×2

MaxPooling lấy giá trị lớn nhất trong cửa sổ $2\times2$:

$$
y_{i,j}
=
\max_{u,v\in\{0,1\}}
x_{2i+u,2j+v}.
$$

Tầng này không có tham số, giảm chiều không gian và chi phí tính toán, đồng thời tăng vùng tiếp nhận. Trong lan truyền ngược, gradient chỉ được truyền về vị trí có giá trị lớn nhất. MaxPooling tạo invariance cục bộ với dịch chuyển nhỏ, nhưng có thể làm mất thông tin vị trí chính xác — điều cần cân nhắc với FER vì vị trí tương đối của mắt, miệng, lông mày khá quan trọng.

### 2.2.5. Dropout

Dropout ngẫu nhiên tắt một tỉ lệ nơ-ron trong huấn luyện. Với xác suất giữ $q=1-p$, mark $m\sim \mathrm{Bernoulli}(q)$, đầu ra được tính theo inverted dropout:

$$
y = \frac{m\odot x}{q}.
$$

Khi suy luận, không dùng mark và không cần chia cho $q$. Dropout giảm hiện tượng đồng thích nghi giữa các nơ-ron, có tác dụng chuẩn hóa và giảm quá khớp.

Trong mã nguồn, `Dropout(0.2)` được đặt sau **mỗi** khối tích chập (kể cả khối đầu) và `Dropout(0.5)` được đặt trước lớp FC cuối. Đây là mức vừa phải: tỉ lệ 0,2 ở các khối tích chập không quá lớn, còn 0,5 ở phần phân loại giúp giảm quá khớp mạnh hơn.

### 2.2.6. Fully-connected và softmax

Sau các khối tích chập, đặc trưng được làm phẳng để tạo vector $h$. Tầng fully-connected tính logits:

$$
z = W h + b, \qquad z\in\mathbb{R}^7.
$$

Softmax chuyển logits thành phân phối xác suất:

$$
p_k = \frac{\exp(z_k)}{\sum_{j=0}^{6}\exp(z_j)}.
$$

Dự đoán cuối cùng là lớp có xác suất lớn nhất. Với FER, lớp Neutral chủ yếu dễ nhầm với Sad; ngoài ra còn nhầm với Angry và Happy nhưng ít hơn. Do đó phần phân loại cần được đánh giá bằng confusion matrix chứ không chỉ accuracy.

### 2.2.7. Kiến trúc 

Kiến trúc trong mã nguồn gồm ba khối tích chập, mỗi khối lặp `n` lần `Conv3x`, sau đó `MaxPool(2)` và `Dropout(0.2)`. Bộ phân loại gồm `Flatten → Linear(256) → ReLU → Dropout(0.5) → Linear(7)`.

```text
48×48×1
  → [Conv3×3 → BN → ReLU] × r1 → MaxPool(2) → Dropout(0.2)
  → [Conv3×3 → BN → ReLU] × r2 → MaxPool(2) → Dropout(0.2)
  → [Conv3×3 → BN → ReLU] × r3 → MaxPool(2) → Dropout(0.2)
  → Flatten → Linear(256) → ReLU → Dropout(0.5) → Linear(7)
```

Ba cấu hình được khảo sát:

| Kiến trúc    | Kênh       | (r1, r2, r3) | BatchNorm | Số tham số |
| -------------- | ----------- | ------------ | --------- | -----------: |
| `small`      | 16–32–64  | (1, 1, 1)    | Có       |      615.399 |
| `medium`     | 32–64–128 | (1, 2, 2)    | Có       |    1.459.719 |
| `small_nobn` | 16–32–64  | (1, 1, 1)    | Không    |      615.175 |

Sau ba lần pooling, kích thước không gian còn $6\times6$. Với `small`, đầu vào của lớp FC là $64\times6\times6=2.304$; với `medium` là $128\times6\times6=4.608$. Nguyên tắc thiết kế là tăng dần số kênh khi giảm kích thước không gian, đặt BatchNorm sau tích chập, ReLU sau BatchNorm, và dùng Dropout ở các vị trí có nguy cơ quá khớp.

---

## 2.3. Kỹ thuật huấn luyện

### 2.3.1. Hàm mục tiêu có chuẩn hóa

Bài toán tối ưu tổng quát có thể viết:

$$
\min_\theta
\frac{1}{N}\sum_{i=1}^{N}
\mathcal{L}\big(f_\theta(x_i), y_i\big)
+
\frac{\lambda}{2}\|\theta\|_2^2.
$$

Số hạng thứ hai là chuẩn hóa L2/weight decay, giúp hạn chế trọng số quá lớn và giảm quá khớp. Mã nguồn dùng Adam hoặc SGD với tham số `weight_decay`; đây là L2 coupled. Trong thực nghiệm, chỉ E5 dùng `weight_decay = 5e-4`.

### 2.3.2. Adam

Adam duy trì trung bình động bậc một và bậc hai của gradient:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2,
$$

với hiệu chỉnh bias:

$$
\hat m_t = \frac{m_t}{1-\beta_1^t},
\qquad
\hat v_t = \frac{v_t}{1-\beta_2^t}.
$$

Cập nhật tham số:

$$
\theta_{t+1}
=
\theta_t
-
\eta
\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

Trong mã nguồn, Adam dùng mặc định PyTorch: $\beta_1=0{,}9$, $\beta_2=0{,}999$, $\epsilon=10^{-8}$. Adam thích nghi learning rate theo từng tham số, phù hợp với gradient thưa và nhiễu. Với CNN trên ảnh, Adam thường hội tụ nhanh trong giai đoạn đầu. Tuy nhiên, Adam có thể kém tổng quát hóa hơn SGD trong một số bài toán, nên cần so sánh thực nghiệm.

### 2.3.3. SGD có momentum và Nesterov

SGD với momentum tích lũy vận tốc:

$$
v_{t+1} = \mu v_t + \eta g_t,
\qquad
\theta_{t+1} = \theta_t - v_{t+1}.
$$

Nesterov cải tiến bằng cách tính gradient tại vị trí nhìn trước:

$$
v_{t+1}
=
\mu v_t
+
\eta \nabla \mathcal{L}(\theta_t - \mu v_t),
\qquad
\theta_{t+1} = \theta_t - v_{t+1}.
$$

Trong mã nguồn, SGD dùng `momentum=0.9`, `nesterov=True`. Momentum giúp vượt qua vùng gradient nhỏ và giảm dao động; Nesterov thường cho hội tụ mượt hơn. SGD + momentum + Nesterov thường tổng quát hóa tốt nhưng nhạy cảm với learning rate và cần schedule hợp lý.

### 2.3.4. OneCycleLR dạng cosine

OneCycleLR gồm hai pha: tăng learning rate từ giá trị nhỏ lên cực đại trong warmup, sau đó giảm theo cosine. Trong pha warmup, learning rate tăng tuyến tính:

$$
\eta_t
=
\eta_{\text{start}}
+
(\eta_{\max}-\eta_{\text{start}})
\frac{t}{T_{\text{warmup}}}.
$$

Trong pha giảm:

$$
\eta_t
=
\eta_{\min}
+
\frac{\eta_{\max}-\eta_{\min}}{2}
\left(
1+\cos\left(\pi\frac{t}{T_{\text{decay}}}\right)
\right).
$$

Momentum thường được điều chỉnh ngược chu kỳ để cân bằng. OneCycleLR giúp hội tụ nhanh, ổn định và có tác dụng chuẩn hóa, nhưng cần đặt tổng số bước chính xác theo số epoch và số batch.

Trong mã nguồn, `sched='cosine'` được ánh xạ tới
`OneCycleLR(max_lr=lr, total_steps=steps)`, với
`steps = epochs * ceil(N_train / batch_size)` và mặc định PyTorch:
`pct_start=0.3` (warmup 30% số bước), `div_factor=25`, `final_div_factor=1e4`,
`anneal_strategy='cos'`. Lưu ý `cycle_momentum` chỉ ảnh hưởng các param group có
momentum (SGD), không ảnh hưởng Adam.

### 2.3.5. Weight decay

Weight decay thêm số hạng phạt:

$$
\mathcal{L}_{\text{reg}}
=
\mathcal{L}
+
\frac{\lambda}{2}\|\theta\|_2^2.
$$

Việc phạt trọng số lớn giúp mô hình đơn giản hơn, giảm quá khớp. Trong mã nguồn, optimizer nhận toàn bộ `model.parameters()`, nên khi `wd > 0` (chỉ E5 dùng `wd=5e-4`), weight decay cũng áp lên bias và tham số BatchNorm. Nếu muốn loại trừbias/BN, cần tách thành các param group riêng. Ngoài ra, Adam trong PyTorch là L2 coupled; AdamW tách riêng weight decay khỏi cập nhật gradient và thường ổn định hơn, nhưng mã nguồn hiện không dùng AdamW.

### 2.3.6. Label smoothing

Với bài toán $K=7$ lớp, nhãn one-hot $y$ được làm mềm:

$$
q_c =
\begin{cases}
1-\varepsilon, & c=y,\\
\dfrac{\varepsilon}{K-1}, & c\ne y.
\end{cases}
$$

Hàm mất mát:

$$
\mathcal{L}
=
-\sum_{c=0}^{K-1}
q_c \log p_\theta(c\mid x).
$$

Label smoothing giảm tự tin quá mức, cải thiện hiệu chuẩn xác suất và giúp mô hình bền vững hơn với nhãn nhiễu. Với FER, nhiều ảnh có biểu cảm mơ hồ hoặc gán nhãnchủ quan, nên label smoothing đặc biệt có ý nghĩa. Trong mã nguồn, E6 dùng`label_smoothing=0.1`.

---

## 2.4. Mất cân bằng lớp và class weights

### 2.4.1. Vấn đề mất cân bằng

FER2013 mất cân bằng mạnh. Lớp Happy có khoảng 7.215 ảnh huấn luyện, trong khi Disgust chỉ có 436 ảnh, chênh lệch hơn 16 lần:

$$
\frac{7215}{436}\approx 16{,}55.
$$

Nếu huấn luyện bằng cross-entropy thông thường, mô hình có xu hướng dự đoán về các lớp đa số. Accuracy có thể cao nhưng recall của lớp thiểu số rất thấp. Vì vậy cần dùng trọng số lớp.

### 2.4.2. Cross-entropy có trọng số

Hàm mất mát có trọng số:

$$
\mathcal{L}
=
-\frac{1}{N}
\sum_{i=1}^{N}
w_{y_i}
\log p_\theta(y_i\mid x_i).
$$

Với $w_{y_i}$ là trọng số của lớp đúng. Gradient theo logits có dạng:

$$
\frac{\partial \mathcal{L}}{\partial z_k}
=
\frac{1}{N}w_y\big(p_k-\mathbf{1}[k=y]\big).
$$

Như vậy, mẫu thuộc lớp thiểu số tạo gradient lớn hơn, buộc mô hình chú ý hơn đếnlớp đó. Trong PyTorch, với `reduction='mean'`, loss có trọng số được chuẩn hóa theo tổng trọng số của các mẫu trong batch; công thức $1/N$ ở trên là dạng chuẩn để phân tích.

### 2.4.3. Hai chế độ chuẩn hóa trọng số

Gọi $N$ là tổng số mẫu huấn luyện, $n_c$ là số mẫu lớp $c$, $K=7$.

- **Chế độ `sqrt`:**

$$
w_c = \sqrt{\frac{N}{K n_c}}.
$$

- **Chế độ `inverse`:**

$$
w_c = \frac{N}{K n_c}.
$$

Sau đó trọng số được chuẩn hóa để trung bình trên các lớp bằng 1:

$$
\tilde w_c
=
\frac{w_c}{\frac{1}{K}\sum_{j=1}^{K}w_j}.
$$

Với $n_{Happy}=7215$ và $n_{Disgust}=436$:

- `inverse`: $w_{Disgust}/w_{Happy}=7215/436\approx16{,}55$, còn
  $w_{Happy}/w_{Disgust}\approx0{,}0604$.
- `sqrt`: $w_{Disgust}/w_{Happy}\approx\sqrt{16{,}55}\approx4{,}07$, còn
  $w_{Happy}/w_{Disgust}\approx0{,}246$.

Như vậy, `inverse` giảm mạnh trọng số của lớp đa số (tương đương tăng mạnh ưu tiên cho lớp thiểu số), có thể khiến mô hình dự đoán quá nhiều về lớp thiểu số. Chế độ `sqrt` làm dịu mức phạt: tỉ lệ trọng số của Disgust so với Happy giảm từ khoảng 16,55 xuống 4,07, cân bằng hơn giữa recall lớp thiểu số và precision toàn cục.

---

## 2.5. Tăng cường dữ liệu và random erasing

### 2.5.1. Mục đích

Tăng cường dữ liệu tạo các biến thể hợp lý của ảnh huấn luyện, giúp mô hình học bất biến với thay đổi nhỏ về tư thế, vị trí và ánh sáng, từ đó giảm quá khớp. Tăng cường chỉ áp dụng cho tập huấn luyện; validation/test phải giữ nguyên phân phối gốc.

### 2.5.2. Các phép biến đổi hình học

- **Xoay ngẫu nhiên $\pm 12^\circ$:** mô phỏng đầu nghiêng nhẹ; biên độ nhỏ để tránh làm méo biểu cảm.
- **Dịch ngang/dọc $\pm 0{,}12$ kích thước ảnh:** mô phỏng khuôn mặt lệch khỏi tâm. Trong mã nguồn, phép dịch được nhân thêm hệ số 2 khi đưa vào `affine_grid`, nên biên độ thực tế đúng bằng 0,12 kích thước ảnh.
- **Zoom ngẫu nhiên:** hệ số `sc = 1 ± 0.12`, sau đó tọa độ được chia cho `sc`; tỉ lệ phóng thực tế là $1/sc \in [0{,}893; 1{,}136]$, không đối xứng hoàn toàn quanh 1.
- **Lật ngang với xác suất 0,5:** với FER, lật ngang thường giữ nguyên cảm xúc và giúp tăng đa dạng dữ liệu.

Các phép này được chọn với biên độ vừa phải vì biến đổi quá mạnh có thể làm mất thông tin biểu cảm, đặc biệt ở vùng mắt và miệng.

### 2.5.3. Random erasing

Với xác suất $p=0{,}25$, chọn một hình chữ nhật ngẫu nhiên có diện tích bằng $2\%$–$10\%$ diện tích ảnh, tỉ lệ cạnh $0{,}3$–$3{,}3$, rồi tô vùng đó bằng 0. Nếu thực hiện sau chuẩn hóa, giá trị 0 tương ứng với giá trị trung bình của ảnh.

Công thức:

chọn diện tích

$$
S_e = H W u,
\qquad u\sim\mathcal{U}(0{,}02;0{,}10),
$$

tỉ lệ cạnh (log-uniform, đúng như mã nguồn)

$$
r_e \sim \exp\big(\mathcal{U}(\log 0{,}3;\log 3{,}3)\big),
\qquad r_e=\frac{h_e}{w_e},
$$

khi đó

$$
h_e=\sqrt{S_e r_e},
\qquad
w_e=\sqrt{\frac{S_e}{r_e}}.
$$

Mã nguồn làm tròn $h_e,w_e$ và cắt theo biên ảnh. Random erasing buộc mô hình không phụ thuộc vào một vùng khuôn mặt cụ thể, tăng tính bền vững khi bị che khuất. Tuy nhiên, nếu xác suất quá cao hoặc vùng xóa quá lớn, mô hình có thể mất thông tin quan trọng và giảm hiệu năng. Vì vậy $p=0{,}25$ và diện tích 2–10% là lựa chọn cân bằng.

---

## 2.6. Hàm mất mát và độ đo đánh giá

### 2.6.1. Cross-entropy

Cross-entropy cơ bản:

$$
\mathcal{L}
=
-\sum_{c=0}^{6}
y_c \log p_\theta(c\mid x),
$$

trong đó $y_c$ là nhãn one-hot. Với nhãn one-hot, công thức rút gọn thành $-\log p_\theta(y\mid x)$. Khi kết hợp label smoothing và class weights, cần cài đặt cẩn thận để không làm sai lệch mục tiêu. Cross-entropy là hàm mất mát phù hợp cho phân loại đa lớp vì nó tối ưu trực tiếp xác suất của lớp đúng.

### 2.6.2. Accuracy

$$
\mathrm{Acc}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}[\hat y_i = y_i].
$$

Accuracy dễ hiểu nhưng không đủ với dữ liệu mất cân bằng: một mô hình dự đoán thiên về lớp Happy vẫn có thể đạt accuracy khá cao trong khi bỏ qua Disgust. Vì vậy cần dùng thêm các độ đo theo lớp.

### 2.6.3. Precision, Recall, F1-score

Với mỗi lớp $c$:

$$
P_c = \frac{TP_c}{TP_c+FP_c},
\qquad
R_c = \frac{TP_c}{TP_c+FN_c},
\qquad
F1_c = \frac{2P_c R_c}{P_c+R_c}.
$$

Macro-F1:

$$
\mathrm{F1}_{\text{macro}}
=
\frac{1}{7}
\sum_{c=0}^{6}F1_c.
$$

Weighted-F1:

$$
\mathrm{F1}_{\text{weighted}}
=
\sum_{c=0}^{6}
\frac{n_c}{N}F1_c.
$$

Macro-F1 đối xử công bằng giữa các lớp, phù hợp với FER2013 mất cân bằng.Weighted-F1 phản ánh hiệu năng tổng thể theo phân phối dữ liệu.

### 2.6.4. Confusion matrix

Ma trận nhầm lẫn $7\times7$ có phần tử

$$
M_{ij}
=
\#\{\text{mẫu thật lớp }i,\ \text{dự đoán lớp }j\}.
$$

Đường chéo là dự đoán đúng; ngoài đường chéo là lỗi. Có thể chuẩn hóa theo hàng:

$$
\tilde M_{ij}
=
\frac{M_{ij}}{\sum_{j}M_{ij}},
$$

để đọc trực tiếp recall của từng lớp. Trong kết quả của mô hình cải tiến, các cặp nhầm lẫn lớn nhất là Sad–Neutral (168), Fear–Angry (105), Fear–Sad (101), Fear–Surprise (78), Neutral–Sad (77), Fear–Neutral (75), Angry–Neutral (70) vàSad–Angry (61).Confusion matrix giúp phát hiện các cặp này để điều chỉnh class weights,augmentation hoặc kiến trúc.

## Kết luận chương

Chương này đã trình bày nền tảng lý thuyết cho mô hình phân loại cảm xúc trên FER2013: từ phát biểu bài toán, vai trò của CNN, các thành phần Conv2D–BatchNorm–ReLU–Pooling–Dropout–FC, đến các kỹ thuật huấn luyện như Adam, SGD momentum, OneCycleLR, weight decay và label smoothing. Đồng thời, chương làm rõ cách xử lý mất cân bằng lớp bằng trọng số cross-entropy, vai trò của tăng cường dữ liệu và random erasing, cùng hệ thống độ đo gồm accuracy, precision, recall, F1 và confusion matrix. Đây là cơ sở để Chương 3 mô tả dữ liệu và tiền xử lý, và Chương 4 trình bày thiết kế thực nghiệm cùng mã nguồn triển khai.

---

# Chương 3 – Dữ liệu và tiền xử lý

## 3.1. Giới thiệu FER2013

FER2013 gồm 35.887 ảnh khuôn mặt xám 48×48, mỗi ảnh thuộc một trong bảy cảm xúc. Dữ liệu được chia thành ba tập theo trường `Usage`:

| Tập            |         Số ảnh | Vai trò                     |
| --------------- | ---------------: | ---------------------------- |
| Training        |           28.709 | Huấn luyện                 |
| PublicTest      |            3.589 | Validation (chọn mô hình) |
| PrivateTest     |            3.589 | Test (đánh giá cuối)     |
| **Tổng** | **35.887** |                              |

## 3.2. Tiền xử lý

Mỗi ảnh được biểu diễn bằng 2.304 giá trị pixel (48×48) trong CSV. Quy trình:

1. Đọc CSV `icml_face_data.csv` (cột `emotion`, `pixels`, `Usage`).
2. Chuyển chuỗi pixel thành mảng `uint8` kích thước 48×48.
3. Lưu vào `fer2013.npz` gồm `X`, `y`, `usage` (script `prepare_data.py`).
4. Khi huấn luyện, chuẩn hóa về khoảng $[-1, 1]$:

$$
x' = \frac{x/255 - 0.5}{0.5}
$$

Việc chuẩn hóa giúp mạng hội tụ ổn định hơn và phù hợp với BatchNorm.

**`prepare_data.py` – đọc CSV và lưu `fer2013.npz`:**

```python
"""Chuyển CSV FER2013 (emotion,pixels,Usage) sang fer2013.npz cho train.py.

Ví dụ: python prepare_data.py --csv ../icml_face_data.csv --out ../fer2013.npz
"""
import argparse
import csv
import os

import numpy as np

from fer.config import BASE_DIR, DATA_NPZ

USAGES = ('Training', 'PublicTest', 'PrivateTest')
EXPECTED = {'Training': 28709, 'PublicTest': 3589, 'PrivateTest': 3589}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Tạo fer2013.npz từ CSV FER2013.')
    p.add_argument('--csv', default=os.path.join(BASE_DIR, 'icml_face_data.csv'))
    p.add_argument('--out', default=DATA_NPZ)
    return p.parse_args(argv)


def read_csv(path):
    X, y, usage = [], [], []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fields = {name.strip().lower(): name for name in (reader.fieldnames or [])}
        missing = {'emotion', 'pixels', 'usage'} - set(fields)
        if missing:
            raise ValueError(f'Thiếu cột {sorted(missing)} trong {path}; có: {reader.fieldnames}')
        for row in reader:
            X.append(np.array(row[fields['pixels']].split(), dtype=np.uint8).reshape(48, 48))
            y.append(int(row[fields['emotion']]))
            usage.append(row[fields['usage']].strip())
    return np.stack(X), np.array(y, dtype=np.int64), np.array(usage)


def main(argv=None):
    a = parse_args(argv)
    X, y, usage = read_csv(a.csv)
    unknown = sorted(set(usage) - set(USAGES))
    if unknown:
        raise ValueError(f'Giá trị Usage lạ: {unknown}')
    counts = {u: int((usage == u).sum()) for u in USAGES}
    print('Số ảnh:', counts)
    for u, n in EXPECTED.items():
        if counts[u] != n:
            print(f'Cảnh báo: {u} có {counts[u]} ảnh, kỳ vọng {n}')
    np.savez_compressed(a.out, X=X, y=y, usage=usage)
    print('Đã lưu', a.out)


if __name__ == '__main__':
    main()
```

**`fer/data.py` – nạp `.npz` và chuẩn hóa về `[-1, 1]`:**

```python
"""Nạp dữ liệu FER2013 đã tiền xử lý (.npz) và chuyển sang tensor."""
import numpy as np
import torch


def load_fer(path):
    d = np.load(path, allow_pickle=True)
    X, y, u = d['X'], d['y'], d['usage']
    sp = lambda name: (X[u == name], y[u == name])
    return sp('Training'), sp('PublicTest'), sp('PrivateTest')


def to_tensor(X):
    return (torch.from_numpy(X).float().unsqueeze(1) / 255.0 - 0.5) / 0.5


def load_splits(path):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = load_fer(path)
    Xtr, Xva, Xte = map(to_tensor, (Xtr, Xva, Xte))
    ytr, yva, yte = map(torch.from_numpy, (ytr, yva, yte))
    return (Xtr, ytr), (Xva, yva), (Xte, yte)
```

## 3.3. Phân bố lớp

| Lớp            |         Training |      PublicTest |     PrivateTest |
| --------------- | ---------------: | --------------: | --------------: |
| Angry           |            3.995 |             467 |             491 |
| Disgust         |              436 |              56 |              55 |
| Fear            |            4.097 |             496 |             528 |
| Happy           |            7.215 |             895 |             879 |
| Sad             |            4.830 |             653 |             594 |
| Surprise        |            3.171 |             415 |             416 |
| Neutral         |            4.965 |             607 |             626 |
| **Tổng** | **28.709** | **3.589** | **3.589** |

![Phân bố lớp](BT02/Code_FER2013/outputs_long_sqrt/figures/class_hist.png)

Nhận xét: Happy chiếm tỉ lệ lớn nhất, Disgust chỉ chiếm khoảng 1,5% tổng dữ liệu. Sự mất cân bằng này là nguyên nhân chính khiến mô hình cơ bản bỏ qua lớp Disgust.

## 3.4. Mẫu dữ liệu

![Mẫu ảnh theo lớp](BT02/Code_FER2013/outputs_long_sqrt/figures/samples.png)

## 3.5. Tăng cường dữ liệu

Ảnh huấn luyện được tăng cường theo từng batch bằng `affine_grid` + `grid_sample`, sau đó áp dụng random erasing:

![Minh họa augmentation](BT02/Code_FER2013/outputs_long_sqrt/figures/augmentation.png)

Tham số mặc định của phiên bản cải tiến:

| Tham số       | Giá trị                                      |
| -------------- | ---------------------------------------------- |
| Xoay           | ±12°                                         |
| Dịch          | ±0,12                                         |
| Zoom           | 0,12                                           |
| Lật ngang     | 50%                                            |
| Random erasing | p = 0,25; diện tích 2–10%; tỉ lệ 0,3–3,3 |

**`fer/augment.py` – biến đổi affine theo batch và random erasing:**

```python
"""Augmentation theo batch: affine grid + random erasing."""
import math

import torch
import torch.nn.functional as F


def random_erase(x, p=0.25, scale=(0.02, 0.10), ratio=(0.3, 3.3)):
    """Tô một vùng chữ nhật ngẫu nhiên bằng 0 (giá trị trung bình sau chuẩn hóa)."""
    if p <= 0:
        return x
    n, _, h, w = x.shape
    device = x.device
    area = h * w
    target = area * (scale[0] + (scale[1] - scale[0]) * torch.rand(n, device=device))
    log_ratio = math.log(ratio[0]) + (math.log(ratio[1]) - math.log(ratio[0])) * torch.rand(n, device=device)
    aspect = torch.exp(log_ratio)
    eh = torch.sqrt(target * aspect).round().long().clamp(1, h)
    ew = torch.sqrt(target / aspect).round().long().clamp(1, w)
    y0 = (torch.rand(n, device=device) * (h - eh + 1)).long()
    x0 = (torch.rand(n, device=device) * (w - ew + 1)).long()
    ys = torch.arange(h, device=device).view(1, h, 1)
    xs = torch.arange(w, device=device).view(1, 1, w)
    mask = (ys >= y0[:, None, None]) & (ys < (y0 + eh)[:, None, None])
    mask = mask & (xs >= x0[:, None, None]) & (xs < (x0 + ew)[:, None, None])
    mask = mask & (torch.rand(n, 1, 1, device=device) < p)
    return x.masked_fill(mask[:, None], 0.0)


def augment(x, rot=12.0, shift=0.12, zoom=0.12, flip=True,
            erase_p=0.25, erase_scale=(0.02, 0.10), erase_ratio=(0.3, 3.3)):
    """Affine (xoay/dịch/zoom/lật) + random erasing, chạy được trên CPU và CUDA."""
    n = x.size(0)
    device = x.device
    ang = (torch.rand(n, device=device) * 2 - 1) * rot * math.pi / 180
    sc = 1 + (torch.rand(n, device=device) * 2 - 1) * zoom
    tx, ty = (torch.rand(n, device=device) * 2 - 1) * shift * 2, (torch.rand(n, device=device) * 2 - 1) * shift * 2
    fl = torch.where(torch.rand(n, device=device) < 0.5, -1.0, 1.0) if flip else torch.ones(n, device=device)
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos * fl, -sin, tx], 1), torch.stack([sin * fl, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    out = F.grid_sample(x, grid, padding_mode='border', align_corners=False)
    return random_erase(out, p=erase_p, scale=erase_scale, ratio=erase_ratio)
```

---

# Chương 4 – Phương pháp và thiết kế thực nghiệm

## 4.1. Kiến trúc mô hình

Mỗi khối tích chập (`conv_block`) gồm `n` lần `[Conv3x3 → (BatchNorm) → ReLU]`, sau đó `MaxPool2x2` và `Dropout(0.2)`. Bộ phân loại gồm `Flatten → Linear → ReLU → Dropout(0.5) → Linear(7)`.

Ba cấu hình được khảo sát:

| Kiến trúc    | Kênh       | Số conv mỗi khối | BatchNorm | Số tham số |
| -------------- | ----------- | ------------------- | --------- | -----------: |
| `small`      | 16–32–64  | 1–1–1             | Có       |      615.399 |
| `medium`     | 32–64–128 | 1–2–2             | Có       |    1.459.719 |
| `small_nobn` | 16–32–64  | 1–1–1             | Không    |      615.175 |

Với đầu vào 48×48, sau ba lần pooling đặc trưng còn 6×6. Bộ phân loại của `small` nhận vector $64 \times 6 \times 6 = 2.304$ chiều.

** `fer/models.py` – `conv_block`, `FERNet` và `build_model`:**

```python
"""Kiến trúc CNN cho FER2013."""
import torch.nn as nn

from .config import ARCH


def conv_block(cin, cout, n, bn=True, drop=0.2):
    L = []
    for k in range(n):
        L.append(nn.Conv2d(cin if k == 0 else cout, cout, 3, padding=1))
        if bn:
            L.append(nn.BatchNorm2d(cout))
        L.append(nn.ReLU(inplace=True))
    L += [nn.MaxPool2d(2), nn.Dropout(drop)]
    return L


class FERNet(nn.Module):
    """CNN: các khối [Conv-(BN)-ReLU]xn + MaxPool + Dropout, rồi Flatten-Dense-Dropout-Dense(7)."""

    def __init__(self, channels=(16, 32, 64), reps=(1, 1, 1), dense=256, bn=True, drop_conv=0.2, drop_fc=0.5):
        super().__init__()
        L, c = [], 1
        for ch, n in zip(channels, reps):
            L += conv_block(c, ch, n, bn, drop_conv)
            c = ch
        self.features = nn.Sequential(*L)
        side = 48 // (2 ** len(channels))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * side * side, dense),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_fc),
            nn.Linear(dense, 7),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(arch='small', **overrides):
    cfg = dict(ARCH[arch])
    cfg.update(overrides)
    return FERNet(**cfg)
```

## 4.2. Tổ chức code

Mã nguồn được tổ chức thành package `fer/` để dễ bảo trì và tái lập:

```text
code/
  fer/
    config.py       # nhãn, kiến trúc, đường dẫn
    data.py         # đọc .npz, chuẩn hóa, chuyển tensor
    augment.py      # affine + random erasing
    device.py       # chọn CPU/CUDA
    models.py       # conv_block, FERNet, build_model
    engine.py       # seed, optimizer, scheduler, train/eval, checkpoint
    metrics.py      # confusion matrix, classification report
    plots.py        # đường cong, confusion matrix, hình dữ liệu
    viz.py          # kernel, activation maximization, feature map
    experiments.py  # E1–E6, chạy thí nghiệm, ghi/đọc nhật ký
  train.py          # CLI huấn luyện
  analyze.py        # CLI phân tích sau huấn luyện
  prepare_data.py   # CSV → fer2013.npz
  fer_lib.py        # shim tương thích import cũ
```

Luồng xử lý:

```text
CSV → prepare_data.py → fer2013.npz
        │
        ▼
data.load_splits ──► engine.fit ──► runs/<tên>.pt + .json
        ▲                │
        │                ▼
   augment.augment   analyze.py ──► experiment_log.csv
        │                            best_report.json
        ▼                            figures/*.png
   models.FERNet
```

**  `fer/engine.py` – vòng huấn luyện, đánh giá và checkpoint:**

```python
"""Vòng huấn luyện, đánh giá và dự đoán dùng chung."""
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .augment import augment


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))


def make_optimizer(model, opt, lr, wd):
    if opt == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)


def make_scheduler(opt, sched, max_lr, total_steps):
    if sched == 'cosine':
        return torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, total_steps=total_steps)
    return None


@torch.no_grad()
def predict(model, X, bs=512, device=None):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        batch = X[i:i + bs]
        if device is not None:
            batch = batch.to(device)
        out.append(model(batch).cpu())
    return torch.cat(out)


def train_one_epoch(model, opt, sched, X, y, bs, do_aug, ls, device=None, class_weights=None):
    model.train()
    weight = class_weights
    if weight is not None and device is not None:
        weight = weight.to(device)
    perm = torch.randperm(len(X))
    tl = tc = 0
    for i in range(0, len(X), bs):
        idx = perm[i:i + bs]
        x, yb = X[idx], y[idx]
        if device is not None:
            x, yb = x.to(device), yb.to(device)
        if do_aug:
            x = augment(x)
        opt.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, yb, weight=weight, label_smoothing=ls)
        loss.backward()
        opt.step()
        if sched:
            sched.step()
        tl += loss.item() * len(yb)
        tc += (out.argmax(1) == yb).sum().item()
    return tl / len(X), tc / len(X)


def evaluate(model, X, y, device=None):
    logits = predict(model, X, device=device)
    loss = F.cross_entropy(logits, y).item()
    acc = (logits.argmax(1) == y).float().mean().item()
    return loss, acc


def fit(model, opt, sched, Xtr, ytr, Xva, yva, epochs, bs, do_aug, ls, tag,
        device=None, class_weights=None, t0=None, log=print):
    if t0 is None:
        t0 = time.time()
    hist, best = [], dict(val_acc=0, epoch=-1)
    for ep in range(epochs):
        train_loss, train_acc = train_one_epoch(model, opt, sched, Xtr, ytr, bs, do_aug, ls,
                                                device=device, class_weights=class_weights)
        val_loss, val_acc = evaluate(model, Xva, yva, device=device)
        r = dict(epoch=ep + 1, train_loss=train_loss, train_acc=train_acc, val_loss=val_loss, val_acc=val_acc,
                 lr=opt.param_groups[0]['lr'], elapsed_min=(time.time() - t0) / 60)
        hist.append(r)
        log(json.dumps(r), flush=True)
        if val_acc > best['val_acc']:
            best = dict(val_acc=val_acc, val_loss=val_loss, epoch=ep + 1)
            torch.save(model.state_dict(), tag + '.pt')
    return hist, best


def load_checkpoint(model, path, device=None):
    model.load_state_dict(torch.load(path, map_location=device if device is not None else 'cpu'))
    return model
```

## 4.3. Chế độ huấn luyện

Notebook hỗ trợ ba chế độ epoch:

| Chế độ              | E1 | E2 | E3 | E4 | E5 | E6 | Mục đích                  |
| ---------------------- | -: | -: | -: | -: | -: | -: | ---------------------------- |
| `quick`              |  2 |  2 |  2 |  2 |  2 |  2 | Chạy thử nhanh             |
| `full`               | 12 | 12 | 12 | 12 | 12 | 10 | Cấu hình báo cáo cơ sở |
| `long` (mặc định) | 20 | 25 | 25 | 25 | 25 | 30 | Cấu hình cải tiến        |

## 4.4. Thiết kế thí nghiệm E1–E6

Sáu thí nghiệm được thiết kế để cô lập ảnh hưởng của từng yếu tố:

| Thí nghiệm           | Kiến trúc |  Aug  | Optimizer |   LR | Scheduler |   WD |  LS | Mục tiêu                              |
| ---------------------- | ----------- | :----: | --------- | ---: | --------- | ---: | --: | --------------------------------------- |
| E1_small_noaug         | small       | Không | Adam      | 1e-3 | none      |    0 |   0 | Baseline không augmentation            |
| E2_small_aug           | small       |  Có  | Adam      | 1e-3 | none      |    0 |   0 | Tác động của augmentation           |
| E3_small_aug_onecycle  | small       |  Có  | Adam      | 3e-3 | cosine    |    0 |   0 | Tác động của OneCycle               |
| E4_small_nobn_aug      | small_nobn  |  Có  | Adam      | 1e-3 | none      |    0 |   0 | Tác động của BatchNorm              |
| E5_small_sgd_onecycle  | small       |  Có  | SGD       | 0,05 | cosine    | 5e-4 |   0 | So sánh SGD với Adam                  |
| E6_medium_aug_onecycle | medium      |  Có  | Adam      | 3e-3 | cosine    |    0 | 0,1 | Kiến trúc lớn hơn + label smoothing |

**  `fer/experiments.py` – khai báo sáu cấu hình E1–E6:**

```python
EXPERIMENTS = [
    dict(name='E1_small_noaug', arch='small', aug=0, epochs=12, long_epochs=20),
    dict(name='E2_small_aug', arch='small', aug=1, epochs=12, long_epochs=25),
    dict(name='E3_small_aug_onecycle', arch='small', aug=1, epochs=12, long_epochs=25,
         sched='cosine', lr=3e-3),
    dict(name='E4_small_nobn_aug', arch='small_nobn', aug=1, epochs=12, long_epochs=25),
    dict(name='E5_small_sgd_onecycle', arch='small', aug=1, epochs=12, long_epochs=25,
         opt='sgd', sched='cosine', lr=0.05, wd=5e-4),
    dict(name='E6_medium_aug_onecycle', arch='medium', aug=1, epochs=10, long_epochs=30,
         sched='cosine', lr=3e-3, ls=0.1),
]
```

## 4.5. Cân bằng lớp

Cấu hình cơ sở dùng `class_weights = none`. Cấu hình cải tiến dùng `sqrt`:

| Chế độ   | Công thức                | Đặc điểm                              |
| ----------- | -------------------------- | ----------------------------------------- |
| `none`    | $w_c = 1$                | Không cân bằng                         |
| `sqrt`    | $w_c = \sqrt{N/(7 n_c)}$ | Cân bằng vừa phải, giảm thiên lệch |
| `inverse` | $w_c = N/(7 n_c)$        | Cân bằng mạnh, dễ nhiễu              |

**  `fer/experiments.py` – tính và chuẩn hóa class weights:**

```python
# Trích run_experiment(): chỉ tính trọng số khi class_weights != 'none'
weight_tensor = None
if class_weights not in (None, 'none'):
    counts = torch.bincount(ytr, minlength=7).float().clamp_min(1)
    if class_weights == 'sqrt':
        weight_tensor = (len(ytr) / (7 * counts)).sqrt()
    elif class_weights in ('inverse', 'balanced'):
        weight_tensor = len(ytr) / (7 * counts)
    else:
        raise ValueError(f'Unknown class_weights: {class_weights}')
    weight_tensor = weight_tensor / weight_tensor.mean()
```

## 4.6. Quản lý kết quả

Để tránh trộn lẫn các cấu hình, notebook ghi kết quả vào thư mục riêng:

```text
Code_FER2013/outputs_<mode>_<weights>/
  runs/                # E1..E6 .json + .pt
  figures/             # toàn bộ hình phân tích
  experiment_log.csv
  best_report.json
  fer2013_results.zip
```

Ví dụ: `outputs_long_sqrt/` (đã có kết quả), `outputs_full_none/`,
`outputs_long_none/`, `outputs_long_inverse/`.

**  `fer/experiments.py` – ghi nhật ký thí nghiệm ra CSV:**

```python
def write_experiment_log(runs, path):
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['Thí nghiệm', 'Kiến trúc', 'Số tham số', 'Augmentation', 'Optimizer', 'LR', 'Scheduler',
                    'Weight decay', 'Label smoothing', 'Class weights', 'Epochs', 'Best epoch',
                    'Train acc @best', 'Val loss @best', 'Val acc (PublicTest)', 'Test acc (PrivateTest)',
                    'Thời gian (phút)'])
        for r in runs:
            a, b = r['args'], r['best']
            h = r['hist'][b['epoch'] - 1]
            w.writerow([r['name'], a['arch'], r['params'], 'có' if a['aug'] else 'không', a['opt'], a['lr'],
                        a['sched'], a['wd'], a['ls'], a.get('class_weights', 'none'), a['epochs'], b['epoch'],
                        f"{h['train_acc']:.4f}", f"{b['val_loss']:.4f}", f"{b['val_acc']:.4f}",
                        f"{r['test_acc_at_best_val']:.4f}", f"{r['minutes']:.1f}"])
```

## 4.7. Môi trường thực nghiệm

- Thư viện: PyTorch, NumPy, scikit-learn, Matplotlib.
- Batch size 64, seed 0, lưu checkpoint tại epoch có validation accuracy tốt nhất.

---

# Chương 5 – Kết quả thực nghiệm và đánh giá

## 5.1. Mô hình gốc (12/10 epoch, không class weights, CPU)

Bảng tổng hợp từ `experiment_log.csv` ở thư mục gốc (kết quả cũ, huấn luyện trên CPU):

| Thí nghiệm           |  Tham số |  Aug  | Optimizer |   LR | Scheduler | Epochs | Best epoch | Train acc | Val loss | Val acc | **Test acc** | Thời gian (phút) |
| ---------------------- | --------: | :----: | --------- | ---: | --------- | -----: | ---------: | --------: | -------: | ------: | -----------------: | -----------------: |
| E1_small_noaug         |   615.399 | Không | Adam      | 1e-3 | none      |     12 |          8 |    0,4876 |   1,2259 |  0,5394 |             0,5442 |                7,1 |
| E2_small_aug           |   615.399 |  Có  | Adam      | 1e-3 | none      |     12 |         12 |    0,4546 |   1,2453 |  0,5230 |             0,5297 |                7,3 |
| E3_small_aug_onecycle  |   615.399 |  Có  | Adam      | 3e-3 | cosine    |     12 |         11 |    0,5028 |   1,1575 |  0,5500 |             0,5623 |                6,4 |
| E4_small_nobn_aug      |   615.175 |  Có  | Adam      | 1e-3 | none      |     12 |         12 |    0,4921 |   1,1845 |  0,5478 |             0,5522 |                5,3 |
| E5_small_sgd_onecycle  |   615.399 |  Có  | SGD       | 0,05 | cosine    |     12 |         11 |    0,4989 |   1,1850 |  0,5430 |             0,5681 |                6,6 |
| E6_medium_aug_onecycle | 1.459.719 |  Có  | Adam      | 3e-3 | cosine    |     10 |         10 |    0,5676 |   1,1044 |  0,5837 |   **0,6013** |               23,9 |

Nhận xét:

- E1 (không augmentation) đạt 54,42%; thêm augmentation (E2) không cải thiện ngay khi chưa có lịch tốc độ học tốt.
- OneCycle (E3) cải thiện rõ rệt so với E2 (56,23% so với 52,97%).
- Bỏ BatchNorm (E4) giảm nhẹ độ chính xác so với E3, cho thấy BN có đóng góp.
- SGD + OneCycle (E5) đạt 56,81%, tương đương/xấp xỉ Adam + OneCycle.
- Mô hình `medium` + OneCycle + label smoothing (E6) đạt kết quả tốt nhất **60,13%**, nhờ dung lượng lớn hơn và huấn luyện ổn định hơn.

### mô hình gốc tốt nhất

Mô hình tốt nhất là: `E6_medium_aug_onecycle` với 10 epoch.

| Lớp                   | Precision | Recall | F1-score | Support |
| ---------------------- | --------: | -----: | -------: | ------: |
| Angry                  |     0,498 |  0,519 |    0,508 |     491 |
| Disgust                |     0,000 |  0,000 |    0,000 |      55 |
| Fear                   |     0,442 |  0,180 |    0,256 |     528 |
| Happy                  |     0,811 |  0,879 |    0,844 |     879 |
| Sad                    |     0,461 |  0,436 |    0,448 |     594 |
| Surprise               |     0,680 |  0,767 |    0,721 |     416 |
| Neutral                |     0,521 |  0,730 |    0,608 |     626 |
| **Macro avg**    |     0,488 |  0,502 |    0,484 |   3.589 |
| **Weighted avg** |     0,578 |  0,601 |    0,578 |   3.589 |

![Confusion matrix – mô hình gốc](BT02/Code_FER2013/figures/confusion_matrix.png)

**Chẩn đoán:** mô hình gốc đạt accuracy khá (60,13%) nhưng phân bố năng lực rất lệch. Disgust không được dự đoán đúng lần nào (F1 = 0), Fear chỉ nhận ra 18% mẫu, trong khi Happy đạt F1 0,844. Nguyên nhân gồm ba yếu tố:

- **Mất cân bằng lớp:** Disgust chỉ có 436 ảnh huấn luyện so với 7.215 ảnh Happy.
- **Huấn luyện ngắn:** 10–12 epoch, mô hình chưa học hết các lớp khó.
- **Augmentation yếu:** xoay ±10°, dịch/zoom 0,1, chưa có random erasing.

Ba quan sát này chính là động lực cho phần cập nhật ở mục 5.2.

## 5.2. Cập nhật huấn luyện dài (`long` + `sqrt`)

Cấu hình cải tiến giữ nguyên kiến trúc và siêu tham số của E1–E6 nhưng:

- Huấn luyện dài hơn: E1 20 epoch, E2–E5 25 epoch, E6 30 epoch.
- Dùng `class_weights = sqrt`.
- Augmentation mạnh hơn kèm random erasing.

Đây là cấu hình `sqrt`; hai cấu hình `long/none` và `long/inverse` được so sánh
đầy đủ ở mục 5.5.

Kết quả từ `outputs_long_sqrt/experiment_log.csv`:

| Thí nghiệm           |  Tham số |  Aug  | Optimizer |   LR | Scheduler | Epochs | Best epoch | Train acc | Val loss | Val acc | **Test acc** | Thời gian (phút) |
| ---------------------- | --------: | :----: | --------- | ---: | --------- | -----: | ---------: | --------: | -------: | ------: | -----------------: | -----------------: |
| E1_small_noaug         |   615.399 | Không | Adam      | 1e-3 | none      |     20 |         20 |    0,4849 |   1,2005 |  0,5400 |             0,5570 |                1,3 |
| E2_small_aug           |   615.399 |  Có  | Adam      | 1e-3 | none      |     25 |         25 |    0,4575 |   1,2201 |  0,5316 |             0,5430 |                2,3 |
| E3_small_aug_onecycle  |   615.399 |  Có  | Adam      | 3e-3 | cosine    |     25 |         24 |    0,5090 |   1,1453 |  0,5626 |             0,5790 |                2,5 |
| E4_small_nobn_aug      |   615.175 |  Có  | Adam      | 1e-3 | none      |     25 |         25 |    0,4963 |   1,1563 |  0,5581 |             0,5740 |                2,6 |
| E5_small_sgd_onecycle  |   615.399 |  Có  | SGD       | 0,05 | cosine    |     25 |         24 |    0,5144 |   1,1354 |  0,5620 |             0,5821 |                2,4 |
| E6_medium_aug_onecycle | 1.459.719 |  Có  | Adam      | 3e-3 | cosine    |     30 |         28 |    0,6118 |   1,0325 |  0,6325 |   **0,6484** |                5,2 |

So sánh trực tiếp với cấu hình cơ sở (cùng thí nghiệm):

| Thí nghiệm           | Test acc cơ sở | Test acc cải tiến |              Δ |
| ---------------------- | ---------------: | ------------------: | --------------: |
| E1_small_noaug         |           54,42% |              55,70% |           +1,28 |
| E2_small_aug           |           52,97% |              54,30% |           +1,33 |
| E3_small_aug_onecycle  |           56,23% |              57,90% |           +1,67 |
| E4_small_nobn_aug      |           55,22% |              57,40% |           +2,18 |
| E5_small_sgd_onecycle  |           56,81% |              58,21% |           +1,40 |
| E6_medium_aug_onecycle |           60,13% |    **64,84%** | **+4,71** |

Nhận xét:

- Cả sáu thí nghiệm đều cải thiện khi tăng số epoch, dùng `sqrt` class weights và augmentation mạnh hơn.
- Mức cải thiện lớn nhất ở E6 (+4,71 điểm phần trăm): mô hình `medium` hưởng lợi nhiều nhất từ huấn luyện dài hơn (30 epoch) và random erasing.
- E1 (không augmentation) vẫn tăng 1,28 điểm, cho thấy huấn luyện dài hơn và cân bằng lớp có tác dụng độc lập.

## 5.3. Đối chiếu trực tiếp mô hình gốc và mô hình cập nhật

### Khác biệt cấu hình

| Thành phần  | Mô hình gốc                          | Mô hình cập nhật                                               |
| ------------- | --------------------------------------- | ------------------------------------------------------------------ |
| Epochs E1     | 12                                      | 20                                                                 |
| Epochs E2–E5 | 12                                      | 25                                                                 |
| Epochs E6     | 10                                      | 30                                                                 |
| Class weights | none                                    | sqrt                                                               |
| Augmentation  | xoay ±10°, dịch/zoom 0,1, lật ngang | xoay ±12°, dịch/zoom 0,12, lật ngang + random erasing p = 0,25 |

### So sánh chỉ số tổng thể

| Chỉ số (E6)        | Mô hình gốc | Mô hình cập nhật |                                 Δ |
| -------------------- | -------------: | -------------------: | ---------------------------------: |
| PrivateTest accuracy |         60,13% |     **64,84%** | **+4,71 điểm phần trăm** |
| Macro F1             |          0,484 |      **0,600** |                   **+0,116** |
| Weighted F1          |          0,578 |      **0,635** |                             +0,057 |
| Disgust F1           |          0,000 |      **0,496** |                   **+0,496** |
| Fear F1              |          0,256 |                0,344 |                             +0,088 |
| Happy F1             |          0,844 |                0,877 |                             +0,033 |

### Phân tích kết quả

- Accuracy tăng 4,71 điểm phần trăm (tương đối **+7,8%**), nhưng macro F1 tăng 0,116 (tương đối **+24%**). Mức tăng macro F1 lớn hơn nhiều so với accuracy cho thấy mô hình đã phân loại các lớp tốt hơn.
- Chỉ riêng Disgust đóng góp khoảng 0,071 trong tổng mức tăng macro F1 0,116 (gần 60%). Nói cách khác, phần lớn tiến bộ về macro F1 đến từ việc cải thiện lớp thiểu số bị bỏ quên trước đó.
- Fear vẫn là lớp khó nhất: recall 0,254, thường bị nhầm sang Angry, Sad và Surprise. Điều này cho thấy cân bằng lớp giúp mô hình chú ý hơn, nhưng không giải quyết được sự chồng lấn biểu cảm.
- Chi tiết F1 theo lớp, confusion matrix và đường cong của cả bốn cấu hình được trình bày ở mục 5.4–5.5; mục 5.5 tách riêng ảnh hưởng của `none`/`sqrt`/`inverse`.

## 5.4. Đường cong huấn luyện

So sánh validation accuracy của bốn cấu hình:

| Cấu hình                      | Đường cong validation                                                              |
| ------------------------------- | ------------------------------------------------------------------------------------- |
| **full / none (cơ sở)** | ![Curve full/none](BT02/Code_FER2013/figures/compare_val.png)                         |
| **long / none**           | ![Curve long/none](BT02/Code_FER2013/outputs_long_none/figures/compare_val.png)       |
| **long / sqrt**           | ![Curve long/sqrt](BT02/Code_FER2013/outputs_long_sqrt/figures/compare_val.png)       |
| **long / inverse**        | ![Curve long/inverse](BT02/Code_FER2013/outputs_long_inverse/figures/compare_val.png) |

Đường cong accuracy/loss của từng thí nghiệm trong cấu hình tốt nhất (`long/sqrt`,
`outputs_long_sqrt/figures/`):

| E1                                                              | E2                                                              | E3                                                              |
| --------------------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------------------- |
| ![E1](BT02/Code_FER2013/outputs_long_sqrt/figures/curve_E1.png) | ![E2](BT02/Code_FER2013/outputs_long_sqrt/figures/curve_E2.png) | ![E3](BT02/Code_FER2013/outputs_long_sqrt/figures/curve_E3.png) |

| E4                                                              | E5                                                              | E6                                                              |
| --------------------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------------------- |
| ![E4](BT02/Code_FER2013/outputs_long_sqrt/figures/curve_E4.png) | ![E5](BT02/Code_FER2013/outputs_long_sqrt/figures/curve_E5.png) | ![E6](BT02/Code_FER2013/outputs_long_sqrt/figures/curve_E6.png) |

## 5.5.  phân tích: ảnh hưởng của class weights

Phần này trả lời **RQ2**: class weights `sqrt` có giúp lớp thiểu số mà không làm
giảm đáng kể accuracy không? Ba cấu hình `long` được chạy với cùng kiến trúc,
epoch, augmentation và seed, chỉ khác `class_weights`: `none`, `sqrt`, `inverse`.
Cấu hình `full`/`none` (mô hình gốc) được giữ làm mốc.

### Test accuracy theo từng thí nghiệm

| Cấu hình                 |     E1 |     E2 |     E3 |     E4 |     E5 |               E6 |
| -------------------------- | -----: | -----: | -----: | -----: | -----: | ---------------: |
| full / none (cơ sở, CPU) | 54,42% | 52,97% | 56,23% | 55,22% | 56,81% |           60,13% |
| long / none (GPU)          | 55,98% | 55,61% | 58,82% | 57,26% | 59,29% |           63,95% |
| long / sqrt (GPU)          | 55,70% | 54,30% | 57,90% | 57,40% | 58,21% | **64,84%** |
| long / inverse (GPU)       | 51,38% | 43,55% | 52,49% | 51,30% | 52,16% |           61,10% |

Nhận xét:

- So với baseline, `long/none` và `long/sqrt` tăng cả sáu thí nghiệm.
- `long/none` nhỉnh hơn `long/sqrt` ở E1–E3 và E5 (các mô hình nhỏ), nhưng thua ở
  E4 và đặc biệt E6 (63,95% so với 64,84%).
- `long/inverse` chỉ tốt hơn baseline ở E6; ở E1–E5, trọng số quá mạnh làm giảm
  mạnh accuracy (thấp nhất 43,55% ở E2).

### Chỉ số E6 theo cấu hình

| Cấu hình     |         Test acc |        Macro F1 |     Weighted F1 | Disgust P |       Disgust R |      Disgust F1 | Fear F1 |
| -------------- | ---------------: | --------------: | --------------: | --------: | --------------: | --------------: | ------: |
| full / none    |           60,13% |           0,484 |           0,578 |     0,000 |           0,000 |           0,000 |   0,256 |
| long / none    |           63,95% |           0,540 |           0,625 |     0,750 |           0,055 |           0,102 |   0,367 |
| long / sqrt    | **64,84%** | **0,600** | **0,635** |     0,443 |           0,564 | **0,496** |   0,344 |
| long / inverse |           61,10% |           0,548 |           0,603 |     0,198 | **0,836** |           0,321 |   0,307 |

### F1 theo từng lớp của E6

| Lớp                | full / none |     long / none |     long / sqrt | long / inverse |
| ------------------- | ----------: | --------------: | --------------: | -------------: |
| Angry               |       0,508 |           0,557 | **0,572** |          0,523 |
| Disgust             |       0,000 |           0,102 | **0,496** |          0,321 |
| Fear                |       0,256 | **0,367** |           0,344 |          0,307 |
| Happy               |       0,844 | **0,878** |           0,877 |          0,860 |
| Sad                 |       0,448 |           0,478 | **0,500** |          0,447 |
| Surprise            |       0,721 |           0,758 | **0,763** |          0,753 |
| Neutral             |       0,608 |           0,641 | **0,646** |          0,626 |
| **Macro avg** |       0,484 |           0,540 | **0,600** |          0,548 |

### Confusion matrix của long/none

![Confusion matrix long/none](BT02/Code_FER2013/outputs_long_none/figures/confusion_matrix.png)

`long/none` gần như không dự đoán Disgust: chỉ 3/55 mẫu Disgust được nhận đúng,
1 mẫu Angry bị dự đoán nhầm thành Disgust, và 52 mẫu Disgust bị bỏ sót. Vì vậy
precision Disgust cao (0,750) nhưng recall gần 0 (0,055). Mô hình vẫn đạt 63,95%
nhờ các lớp đa số (Happy 0,878, Neutral 0,641, Surprise 0,758).

### Confusion matrix của long/inverse

![Confusion matrix long/inverse](BT02/Code_FER2013/outputs_long_inverse/figures/confusion_matrix.png)

`long/inverse` đi theo hướng ngược lại: nhận đúng 46/55 mẫu Disgust (recall 0,836)
nhưng dự đoán Disgust cho rất nhiều mẫu khác — 49 Angry, 52 Fear, 41 Sad, 18 Happy,
18 Neutral và 8 Surprise — tạo ra 186 false positive. Precision chỉ còn 0,198 và
F1 0,321. Đây là nguyên nhân chính khiến accuracy giảm còn 61,10% và weighted F1
xuống 0,603.

### Disgust: TP/FP/FN của bốn cấu hình

| Cấu hình     | TP |  FP | FN | Precision | Recall |    F1 |
| -------------- | -: | --: | -: | --------: | -----: | ----: |
| full / none    |  0 |   0 | 55 |     0,000 |  0,000 | 0,000 |
| long / none    |  3 |   1 | 52 |     0,750 |  0,055 | 0,102 |
| long / sqrt    | 31 |  39 | 24 |     0,443 |  0,564 | 0,496 |
| long / inverse | 46 | 186 |  9 |     0,198 |  0,836 | 0,321 |

Bảng này cho thấy bốn kiểu hành vi: bỏ qua hoàn toàn (`full/none`), thận trọng quá
mức (`long/none`), cân bằng (`long/sqrt`) và quá hung hăng (`long/inverse`).

### Phân tích trade-off

- **`none` – under-correct:** gần như không dự đoán Disgust; accuracy vẫn khá
  (63,95%) nhưng macro F1 thấp (0,540). Đây là minh chứng cho việc accuracy có
  thể che giấu thiên lệch lớp.
- **`sqrt` – cân bằng:** accuracy, macro F1 và weighted F1 đều cao nhất
  (64,84% / 0,600 / 0,635); Disgust F1 0,496.
- **`inverse` – over-correct:** recall Disgust cao nhưng precision thấp, kéo
  accuracy xuống 61,10% và weighted F1 xuống 0,603; chi tiết ở phần confusion
  matrix phía trên.

**Theo dung lượng mô hình:** ở các mô hình nhỏ (E1–E5), `long/inverse` giảm mạnh
accuracy (thấp nhất 43,55% ở E2) vì trọng số Disgust quá lớn so với dung lượng;
chỉ E6 medium mới phục hồi phần nào (61,10%). Ngược lại, `long/none` ổn định hơn ở
mô hình nhỏ (E1–E3/E5 cao hơn `long/sqrt`), nhưng macro F1 vẫn thấp vì bỏ rơi
Disgust.

**Kết luận:** `sqrt` là lựa chọn tốt nhất; `none` under-correct, `inverse`
over-correct. Mức tối ưu có thể nằm giữa `sqrt` và `inverse`, ví dụ
$w_c \propto n_c^{-p}$ với $p \in (0{,}5;1)$, nhưng chưa được khảo sát.

### Kiểm tra “better replacement”

Đã so sánh cả ba cấu hình `long` với mô hình gốc. Không cấu hình nào vượt
`long/sqrt` trên đồng thời accuracy, macro F1 và weighted F1:

- `long/none` thắng ở E1–E3/E5 và F1 của Fear/Happy, nhưng thua E6 và macro F1
  (0,540 so với 0,600).
- `long/inverse` chỉ thắng recall Disgust (0,836); thua `long/sqrt` ở cả 7 lớp
  (F1), accuracy, macro F1 và weighted F1.
- `long/sqrt` giữ vị trí tốt nhất ở 5/7 lớp (Angry, Disgust, Sad, Surprise,
  Neutral) và cả ba chỉ số tổng thể.

Vì vậy, mô hình chính vẫn là `E6_medium_aug_onecycle` với `long/sqrt`; không cần
thay thế.

**Lưu ý val/test:** E6 `long/none` có val acc 63,75% (PublicTest) cao hơn
`long/sqrt` 63,25%, nhưng test acc lại thấp hơn (63,95% so với 64,84%). Điều này
cho thấy chỉ chọn mô hình theo val acc có thể bỏ lỡ mô hình tổng quát hóa tốt hơn
trên PrivateTest; cần xem thêm macro F1 và test acc.

> Ghi chú số liệu: E6 `long/inverse` ghi 61,05% trong `experiment_log.csv`và 61,10% trong `best_report.json`; báo cáo dùng 61,10% cho nhất quán với bảng per-class.

## 5.6. Phân tích mô hình tốt nhất

Mô hình tốt nhất ở cấu hình cải tiến: `E6_medium_aug_onecycle`, PrivateTest accuracy **64,84%** (tăng 4,71 điểm phần trăm so với cấu hình cơ sở). Kiểm tra thay thế ở mục 5.5 cho thấy không cấu hình nào vượt `long/sqrt`; đây vẫn là mô hình chính.

![Confusion matrix](BT02/Code_FER2013/outputs_long_sqrt/figures/confusion_matrix.png)

Báo cáo phân loại theo lớp:

| Lớp                   | Precision | Recall | F1-score | Support |
| ---------------------- | --------: | -----: | -------: | ------: |
| Angry                  |     0,551 |  0,595 |    0,572 |     491 |
| Disgust                |     0,443 |  0,564 |    0,496 |      55 |
| Fear                   |     0,534 |  0,254 |    0,344 |     528 |
| Happy                  |     0,873 |  0,882 |    0,877 |     879 |
| Sad                    |     0,524 |  0,478 |    0,500 |     594 |
| Surprise               |     0,722 |  0,810 |    0,763 |     416 |
| Neutral                |     0,564 |  0,757 |    0,646 |     626 |
| **Macro avg**    |     0,601 |  0,620 |    0,600 |   3.589 |
| **Weighted avg** |     0,643 |  0,648 |    0,635 |   3.589 |

Nhận xét:

- **Happy** vẫn là lớp dễ nhất (F1 = 0,877) nhờ đặc trưng miệng cười rõ và dữ liệu nhiều.
- **Disgust** được cải thiện rõ rệt: F1 từ 0 lên **0,496** (precision 0,443, recall 0,564). Class weights `sqrt` giúp mô hình không còn bỏ qua lớp này, dù support test chỉ có 55 mẫu.
- **Fear** vẫn là lớp khó nhất về recall (0,254 dù F1 tăng lên 0,344): mô hình thường nhầm Fear sang Angry, Sad, Surprise và Neutral.
- **Neutral** có recall cao (0,757), precision 0,564; **Surprise** đạt F1 0,763.

Các cặp nhầm lẫn lớn nhất trên PrivateTest:

| Thực tế → Dự đoán | Số mẫu |
| ----------------------- | -------: |
| Sad → Neutral          |      168 |
| Fear → Angry           |      105 |
| Fear → Sad             |      101 |
| Fear → Surprise        |       78 |
| Neutral → Sad          |       77 |
| Fear → Neutral         |       75 |
| Angry → Neutral        |       70 |
| Sad → Angry            |       61 |

So với cấu hình cơ sở, hầu hết nhầm lẫn lớn đều giảm: Sad → Neutral 170 → 168, Fear → Neutral 100 → 75, Angry → Neutral 90 → 70, Neutral → Sad 88 → 77, Angry → Sad 75 → 58. Tuy nhiên Fear → Angry vẫn giữ nguyên 105 và Fear → Sad còn 101, phản ánh sự chồng lấn biểu cảm và đặc trưng mờ của ảnh 48×48.

**  `analyze.py` – nạp lại checkpoint tốt nhất và đánh giá trên PrivateTest:**

```python
best = best_run(runs)
(Xtr, ytr), (Xva, yva), (Xte, yte) = load_fer(a.data)
model = FERNet(**ARCH[best['args']['arch']])
load_checkpoint(model, os.path.join(a.runs, best['name'] + '.pt'))
model.eval()
Xt = to_tensor(Xte)
logits = predict(model, Xt)
pred = logits.argmax(1).numpy()
acc = (pred == yte).mean()

cm = confusion(yte, pred)
save_confusion_matrix(cm, best['name'], acc, os.path.join(a.figures, 'confusion_matrix.png'))
rep = report_dict(yte, pred)
json.dump(dict(best=best['name'], test_acc=float(acc), report=rep, cm=cm.tolist()),
          open(a.report, 'w'), indent=1)
```

**  `fer/metrics.py` – confusion matrix và classification report:**

```python
"""Chỉ số đánh giá: confusion matrix và classification report."""
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from .config import CLASSES


def confusion(y_true, y_pred):
    return sk_confusion_matrix(y_true, y_pred)


def report_dict(y_true, y_pred, classes=CLASSES):
    return classification_report(y_true, y_pred, target_names=classes, output_dict=True)


def report_text(y_true, y_pred, classes=CLASSES):
    return classification_report(y_true, y_pred, target_names=classes)
```

## 5.7. Phân tích lỗi

Phần này trả lời **RQ2**: các nhầm lẫn còn lại tập trung ở đâu và vì sao.

![Dự đoán sai](BT02/Code_FER2013/outputs_long_sqrt/figures/wrong_cases.png)

![Dự đoán đúng](BT02/Code_FER2013/outputs_long_sqrt/figures/true_cases.png)

Nhiều ảnh dự đoán sai có biểu cảm mơ hồ hoặc chất lượng thấp; một số ảnh ngay cả người nhìn cũng khó phân biệt. Điều này cho thấy trần hiệu năng của mô hình huấn luyện từ đầu trên FER2013 là có giới hạn.

## 5.8. Trực quan hóa đặc trưng

Kernel của tầng tích chập đầu tiên học các bộ lọc cạnh/hướng:

![Kernel conv đầu](BT02/Code_FER2013/outputs_long_sqrt/figures/filters_first_conv.png)

Kernel tầng conv cuối (trung bình theo kênh vào):

![Kernel conv cuối](BT02/Code_FER2013/outputs_long_sqrt/figures/filters_last_conv_weights.png)

Activation maximization cho 32 filter đầu của conv cuối:

![Activation maximization](BT02/Code_FER2013/outputs_long_sqrt/figures/filters_last_conv_actmax.png)

Feature map của conv cuối cho một ảnh Happy:

![Feature map](BT02/Code_FER2013/outputs_long_sqrt/figures/feature_maps_last_conv.png)

Các hình cho thấy mạng học được các mẫu cục bộ (mắt, miệng, vùng má) và kích hoạt mạnh ở các vùng có cấu trúc biểu cảm.

**  `fer/viz.py` – kernel, activation maximization và feature map:**

```python
"""Trực quan hóa kernel, activation maximization và feature map."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def conv_layers(model):
    return [m for m in model.features if isinstance(m, nn.Conv2d)]


def show_kernels(conv, path, title, nmax=32):
    W = conv.weight.detach().numpy()
    if W.shape[1] == 1:
        fig, axs = plt.subplots(4, 8, figsize=(10, 5.4))
        for k, ax in enumerate(axs.ravel()):
            ax.axis('off')
            if k < W.shape[0]:
                ax.imshow(W[k, 0], cmap='bwr', vmin=-abs(W).max(), vmax=abs(W).max())
    else:
        M = W.mean(1)
        fig, axs = plt.subplots(4, 8, figsize=(10, 5.4))
        for k, ax in enumerate(axs.ravel()):
            ax.axis('off')
            if k < min(nmax, M.shape[0]):
                ax.imshow(M[k], cmap='bwr', vmin=-np.abs(M).max(), vmax=np.abs(M).max())
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def feature_extractor(model):
    last = conv_layers(model)[-1]
    idx_last = list(model.features).index(last)
    has_bn = idx_last + 2 < len(model.features) and isinstance(model.features[idx_last + 1], nn.BatchNorm2d)
    upper = idx_last + 3 if has_bn else idx_last + 2
    sub = nn.Sequential(*list(model.features)[:upper])
    for p_ in sub.parameters():
        p_.requires_grad_(False)
    sub.eval()
    return sub


def activation_maximization(sub, path):
    torch.manual_seed(0)
    nf = 32
    x = (torch.randn(nf, 1, 48, 48) * 0.1).requires_grad_(True)
    opt = torch.optim.Adam([x], lr=0.05)
    for _ in range(150):
        opt.zero_grad()
        a = sub(x)
        loss = -a[torch.arange(nf), torch.arange(nf)].mean() + 1e-3 * (x ** 2).mean()
        loss.backward()
        opt.step()
    fig, axs = plt.subplots(4, 8, figsize=(11, 6))
    for k, ax in enumerate(axs.ravel()):
        ax.imshow(x[k, 0].detach().numpy(), cmap='gray')
        ax.axis('off')
        ax.set_title(f'f{k}', fontsize=7)
    fig.suptitle('Activation maximization: mẫu đầu vào kích hoạt mạnh nhất 32 filter đầu của conv cuối')
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def save_feature_maps(sub, Xt, Xte, i0, path):
    fm = sub(Xt[i0:i0 + 1])[0].detach().numpy()
    fig, axs = plt.subplots(4, 9, figsize=(12, 5.6))
    axs[0, 0].imshow(Xte[i0], cmap='gray')
    axs[0, 0].set_title('Input (Happy)', fontsize=7)
    for ax in axs.ravel():
        ax.axis('off')
    for k in range(31):
        axs.ravel()[k + 1].imshow(fm[k], cmap='viridis')
        axs.ravel()[k + 1].set_title(f'f{k}', fontsize=6)
    fig.suptitle('Feature map của conv cuối (6x6) cho một ảnh Happy')
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
```

## 5.9. Thảo luận

- **Huấn luyện dài hơn + random erasing** giúp `long/none` và `long/sqrt` tăng cả sáu thí nghiệm (1,28–4,71 điểm phần trăm); `long/inverse` chỉ tăng ở E6, còn E1–E5 giảm mạnh do trọng số quá lớn.
- **OneCycle** vẫn là yếu tố quan trọng: E3/E5/E6 (dùng cosine OneCycle) đạt kết quả cao hơn E2/E4 (không dùng).
- **BatchNorm** đóng góp tích cực: E3 (có BN) đạt 57,90% so với E4 (không BN) 57,40% trong cùng điều kiện.
- **Kiến trúc medium** + label smoothing tiếp tục cho kết quả tốt nhất (**64,84%**), vượt E5 (58,21%) gần 6,6 điểm phần trăm.
- **Class weights quyết định hiệu năng lớp thiểu số:** `none` để Disgust F1 = 0,102 (recall 0,055); `sqrt` đưa lên 0,496 (precision 0,443, recall 0,564); `inverse` đẩy recall lên 0,836 nhưng precision chỉ 0,198, F1 0,321.
- **`sqrt` là điểm cân bằng tốt nhất:** accuracy 64,84%, macro F1 0,600, weighted F1 0,635 — cao nhất trong bốn cấu hình.
- **`none` và `inverse` minh họa hai kiểu sai:** `none` under-correct (bỏ rơi Disgust), `inverse` over-correct (dự đoán Disgust quá nhiều, kéo accuracy xuống 61,10%). Mức tối ưu có thể nằm giữa `sqrt` và `inverse`, ví dụ $w_c \propto n_c^{-p}$ với $p \in (0,5;1)$.
- **Cơ chế thất bại:** `none` chỉ dự đoán Disgust 4 lần (TP = 3, FP = 1, FN = 52); `inverse` dự đoán Disgust 232 lần (TP = 46, FP = 186, FN = 9), trong đó 49 Angry, 52 Fear và 41 Sad bị kéo sang Disgust.
- **Mất cân bằng lớp và chồng lấn biểu cảm** vẫn là hạn chế chính; Fear recall (0,254) còn thấp, cần focal loss/oversampling hoặc transfer learning để cải thiện thêm.

---

# Chương 6 – Kết luận và hướng phát triển

## 6.1. Kết luận

Bài tập lớn đã xây dựng thành công quy trình hoàn chỉnh cho bài toán FER2013: tiền xử lý dữ liệu, huấn luyện CNN, đánh giá, phân tích lỗi và trực quan hóa đặc trưng. Kết quả cho thấy:

- OneCycle, BatchNorm và kiến trúc lớn hơn giúp cải thiện độ chính xác.
- Cấu hình cơ sở đạt **60,13%**; cấu hình cải tiến (`long` + `sqrt` + random erasing) đạt **64,84%** trên PrivateTest (+4,71 điểm phần trăm), F1 macro tăng từ 0,484 lên 0,600.
-  phân tích class weights: `long/none` đạt 63,95% (macro F1 0,540), `long/sqrt` đạt 64,84% (0,600), `long/inverse` đạt 61,10% (0,548). `sqrt` là lựa chọn tốt nhất; `none` bỏ rơi Disgust (F1 = 0,102), còn `inverse` over-correct (recall 0,836 nhưng precision 0,198).
- Happy là lớp dễ nhất (F1 = 0,877). Class weights `sqrt` giúp Disgust từ F1 = 0 lên **0,496** và Fear từ 0,256 lên 0,344.
- Fear vẫn là lớp khó nhất về recall (0,254), cho thấy bài toán chồng lấn biểu cảm chưa được giải quyết triệt để.

## 6.2. Hạn chế

- Dữ liệu mất cân bằng mạnh; Disgust đã cải thiện nhưng chỉ có 55 mẫu test, còn Fear recall vẫn thấp (0,254).
- Ảnh độ phân giải thấp, nhiều biểu cảm mơ hồ và chồng lấn (Sad/Neutral,Fear/Angry).
- Huấn luyện từ đầu nên trần hiệu năng thấp hơn transfer learning.
-  phân tích `long`/`none` và `long`/`inverse` đã chạy, nhưng chưa khảo sát các mức trọng số trung gian (ví dụ $p=0,75$) hoặc focal loss để tối ưu hơn nữa trade-off precision–recall của Disgust.

## 6.3. Hướng phát triển

- Áp dụng transfer learning (VGG16, ResNet, EfficientNet) với fine-tuning.
- Dùng focal loss, oversampling, hoặc tinh chỉnh class weights dạng $w_c \propto n_c^{-p}$ với $p \in (0,5;1)$ cho lớp thiểu số.
- Ensemble nhiều mô hình và test-time augmentation.
- Tối ưu thêm trên GPU (mixed precision, batch size lớn) và tìm kiếm siêu tham số
  tự động.
- Triển khai demo real-time (webcam) và đánh giá trên dữ liệu thực tế.

---

---

# Phụ lục

## Phụ lục A – Cấu trúc mã nguồn

```text
BT02/
  icml_face_data.csv          # dữ liệu gốc (không commit)
  fer2013.npz                 # dữ liệu đã tiền xử lý
  Code_FER2013/
    code/
      fer/                    # package mô hình & huấn luyện
      train.py                # CLI huấn luyện
      analyze.py              # CLI phân tích
      prepare_data.py         # CSV → npz
      run_all.sh              # chạy E1–E6
      requirements.txt
    FER2013_Colab.ipynb       # notebook local
    runs/                     # checkpoint + JSON cấu hình cơ sở
    figures/                  # hình phân tích cấu hình cơ sở
    experiment_log.csv        # nhật ký cấu hình cơ sở
    best_report.json          # báo cáo cấu hình cơ sở
    outputs_long_sqrt/        # kết quả cấu hình cải tiến (long + sqrt)
      figures/                # hình phân tích cấu hình cải tiến
      experiment_log.csv
      best_report.json
      fer2013_results.zip
```

## Phụ lục B – Notebook

`FER2013_Colab.ipynb` chạy từ thư mục `Code_FER2013/`, huấn luyện E1–E6, chạy phân tích và đóng gói kết quả.
Các biến cấu hình chính:

| Biến             | Giá trị                         | Ý nghĩa                                |
| ----------------- | --------------------------------- | ---------------------------------------- |
| `EPOCHS_MODE`   | `long` / `full` / `quick`   | Chế độ số epoch                      |
| `CLASS_WEIGHTS` | `none` / `sqrt` / `inverse` | Cân bằng lớp                          |
| `FORCE_RERUN`   | `True`/`False`                | Huấn luyện lại dù đã có kết quả |
| `RUN_ANALYSIS`  | `True`/`False`                | Chạy phân tích sau huấn luyện       |
| `OUT_DIR`       | `outputs_<mode>_<weights>/`     | Thư mục kết quả                      |

Notebook tự so sánh Test acc cũ và mới, hiển thị hình và đóng gói `fer2013_results.zip`.

## Phụ lục D – Đối chiếu thư mục kết quả

| Thư mục                      | Chế độ | Class weights | Trạng thái | Ghi chú                                                   |
| ------------------------------ | --------- | ------------- | ------------ | ---------------------------------------------------------- |
| `runs/`, `figures/` (gốc) | full      | none          | Đã có     | Kết quả cơ sở (best 60,13%)                            |
| `outputs_long_sqrt/`         | long      | sqrt          | Đã có     | Kết quả cải tiến (best 64,84%), dùng trong Chương 5 |
|                                |           |               |              |                                                            |
| `outputs_long_none/`         | long      | none          | Đã có     |  phân tích, best 63,95%, macro F1 0,540                      |
| `outputs_long_inverse/`      | long      | inverse       | Đã có     |  phân tích, best 61,10%, macro F1 0,548                      |
|                                |           |               |              |                                                            |

Kết quả trong `outputs_long_none/`, `outputs_long_sqrt/` và `outputs_long_inverse/` gồm `figures/`, `experiment_log.csv`, `best_report.json` và `fer2013_results.zip`; các file checkpoint `.pt`/`.json` riêng lẻ nằm trong gói zip hoặc được tái tạo khi chạy lại notebook. Sau khi so sánh ba cấu hình `long`, `outputs_long_sqrt/` được chọn làm mô hình chính (64,84%, macro F1 0,600).
