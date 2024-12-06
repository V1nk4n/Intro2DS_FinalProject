import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv('anime_preprocessing.csv')
    return data

# Hàm xử lý đề xuất
def recommend_anime(user_input, data, top_n=10):
    # Kiểm tra dữ liệu đầu vào
    if data.empty:
        return pd.DataFrame()  # Trả về DataFrame rỗng nếu dữ liệu đầu vào rỗng

    # Kết hợp các cột thành một chuỗi đặc trưng
    data['combined_features'] = (
        data['Genres'].fillna("") + " " +
        data['Studios'].fillna("") + " " +
        data['Producers'].fillna("") + " " +
        data['Source'].fillna("")
    )
    
    # Kiểm tra xem cột 'combined_features' có rỗng không
    if data['combined_features'].str.strip().eq("").all():
        return pd.DataFrame()  # Trả về DataFrame rỗng nếu tất cả giá trị là trống

    # Sử dụng TF-IDF để mã hóa
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(data['combined_features'])
    except ValueError:
        return pd.DataFrame()  # Trả về DataFrame rỗng nếu xảy ra lỗi
    
    # Tính toán độ tương đồng cosine
    user_vector = vectorizer.transform([user_input])
    user_similarity = cosine_similarity(user_vector, tfidf_matrix).flatten()
    
    # Sắp xếp và lấy ra top_n phim
    similar_indices = user_similarity.argsort()[-top_n:][::-1]
    recommendations = data.iloc[similar_indices]
    return recommendations


# Load dữ liệu
anime_data = load_data()

# Giao diện Streamlit
st.title("Hệ thống đề xuất Anime")
st.write("Chọn ít nhất một trong các thuộc tính sau để tìm phim tương tự!")

# Lựa chọn đầu vào từ người dùng
title = st.text_input("Nhập tiêu đề (Title):", placeholder="Ví dụ: Attack on Titan")
genres = st.selectbox("Nhập thể loại (Genres):", [""] + list(anime_data['Genres'].unique()))
studios = st.selectbox("Nhập studio (Studios):", [""] + list(anime_data['Studios'].unique()))
producers = st.selectbox("Nhập nhà sản xuất (Producers):", [""] + list(anime_data['Producers'].unique()))
source = st.selectbox("Chọn nguồn tài liệu (Source):", [""] + list(anime_data['Source'].unique()))
min_score = st.slider("Chọn điểm số tối thiểu:", min_value=0.0, max_value=10.0, step=0.1)
max_episodes = st.slider("Số tập tối đa:", min_value=1, max_value=4000, step=1)
min_scored_by = st.slider("Số lượng đánh giá tối thiểu:", min_value=0, max_value=3000000, step=1000)

# Nút để bắt đầu tìm kiếm
if st.button("Đề xuất"):
    if not title.strip() and genres == "" and studios == "" and producers == "" and source == "":
        st.warning("Bạn cần nhập ít nhất một trường để hệ thống đề xuất chính xác!")
    else:
        # Tạo chuỗi đầu vào người dùng
        user_input_parts = [field for field in [title, genres, studios, producers, source] if field.strip()]
        user_input = " ".join(user_input_parts) if user_input_parts else 'UNKNOWN'
        
        # Lọc dữ liệu theo tiêu chí
        filtered_data = anime_data[
            (anime_data['Score'] >= min_score) &
            (anime_data['Episodes'] <= max_episodes) &
            (anime_data['Scored By'] >= min_scored_by)
        ]
        
        # Nếu người dùng nhập tiêu đề, ưu tiên lọc theo tiêu đề trước
        if title.strip():
            filtered_data = filtered_data[filtered_data['Title'].str.contains(title, case=False, na=False)]
        
        # Gợi ý phim dựa trên dữ liệu đã lọc
        recommendations = recommend_anime(user_input, filtered_data)
        
        if recommendations.empty:
            st.warning("Không tìm thấy phim nào phù hợp! Vui lòng thay đổi tiêu chí.")
        else:
            st.write("Phim được đề xuất:")
            for _, row in recommendations.iterrows():
                st.write(f"""
                    **{row['Title']}**  
                    - Thể loại: {row['Genres']}  
                    - Studio: {row['Studios']}  
                    - Nhà sản xuất: {row['Producers']}  
                    - Nguồn gốc: {row['Source']}  
                    - Điểm số: {row['Score']}  
                    - Số lượt đánh giá: {row['Scored By']}  
                    - Số tập: {row['Episodes']}  
                """)
