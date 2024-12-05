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
    # Kết hợp các cột thành một chuỗi đặc trưng
    data['combined_features'] = (
        data['Genres'] + " " +
        data['Studios'] + " " +
        data['Producers'] + " " +
        data['Source']
    )
    
    # Sử dụng TF-IDF để mã hóa
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(data['combined_features'])
    
    # Tính toán độ tương đồng cosine
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Tìm phim tương tự dựa trên tiêu chí của người dùng
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
st.write("Chọn các thuộc tính để tìm phim tương tự!")

# Lựa chọn đầu vào từ người dùng
title = st.text_input("Nhập tiêu đề (Title):", placeholder="Ví dụ: Attack on Titan")
genres = st.text_input("Nhập thể loại (Genres):", placeholder="Ví dụ: Action, Comedy")
studios = st.text_input("Nhập studio (Studios):", placeholder="Ví dụ: Madhouse, Kyoto Animation")
producers = st.text_input("Nhập nhà sản xuất (Producers):", placeholder="Ví dụ: Aniplex, Toei Animation")
source = st.selectbox("Chọn nguồn tài liệu (Source):", anime_data['Source'].unique())
min_score = st.slider("Chọn điểm số tối thiểu:", min_value=0.0, max_value=10.0, step=0.1)
max_episodes = st.slider("Số tập tối đa:", min_value=1, max_value=200, step=1)
min_scored_by = st.slider("Số lượng đánh giá tối thiểu:", min_value=0, max_value=int(anime_data['Scored By'].max()), step=1000)

# Nút để bắt đầu tìm kiếm
if st.button("Đề xuất"):
    if not title and not genres and not studios and not producers:
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
        if title:
            filtered_data = filtered_data[filtered_data['Title'].str.contains(title, case=False, na=False)]
        
        # Gợi ý phim dựa trên dữ liệu đã lọc
        recommendations = recommend_anime(user_input, filtered_data)
        
        if not recommendations.empty:
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
                """)
        else:
            st.write("Không tìm thấy phim nào phù hợp!")
