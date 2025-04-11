# Hệ thống nhận diện khuôn mặt sử dụng ```facenet-pytorch``` và ```faiss``` để trích xuất và tìm kiếm đặc trưng khuôn mặt trong thời gian thực.

## Công Nghệ Sử Dụng

- Nhận diện khuôn mặt: Sử dụng ```MTCNN``` từ ```facenet-pytorch``` để phát hiện khuôn mặt trong ảnh/video.

- Trích xuất đặc trưng: Sử dụng mô hình ```InceptionResnetV1``` (pretrained trên vggface2) để trích xuất embedding 512 chiều của khuôn mặt.

- So sánh và tìm kiếm: Sử dụng ```faiss.IndexFlatL2``` để lưu trữ và tìm kiếm các embeddings dựa trên khoảng cách L2.

## Cài Đặt

1. Cài đặt các thư viện

Chạy lệnh sau để cài đặt các thư viện từ requirements.txt:
```
pip install -r requirements.txt
```

2. Cấu Trúc Dữ Liệu

Hệ thống yêu cầu dữ liệu embeddings của khuôn mặt được lưu trữ trong thư mục:

``` /home/jetson/face_recognition/embedding/ ```

Mỗi thư mục con bên trong chứa các tệp JSON của người dùng với định dạng:
```
[UserCode] Username/
    ├── face_1.json
    ├── face_2.json
    ├── ...
```
Mỗi tệp JSON chứa vector đặc trưng khuôn mặt của người đó. (512D)

3. Thêm người dùng từ thư mục embeddings

Trước khi chạy nhận diện, hệ thống sẽ tự động tải dữ liệu từ thư mục embeddings.

4. Chạy chương trình nhận diện khuôn mặt
```
python main.py
```

## Ghi Nhận Lịch Sử Nhận Diện

Khi một người dùng được nhận diện, thông tin sẽ được lưu vào tệp CSV:

``` recognized_users.csv ```

Thông tin bao gồm:

- ID: Mã người dùng

- Username: Tên người dùng

- Timestamp (GMT+7): Thời gian nhận diện

