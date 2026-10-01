"""
chatbot_engine.py - In-House NLP Chatbot Engine for CineVector (CineBot).
Xây dựng 100% từ con số 0:
1. Intent Classification: TF-IDF Vectorizer + Cosine Similarity Intent Matching.
2. Entity Extraction: N-gram & Levenshtein / Fuzzy Matching cho tên phim và thể loại.
3. Conversational Generator: Sinh phản hồi tự nhiên, giải thích toán học và sinh payload thẻ phim.
Không phụ thuộc bất kỳ API bên thứ ba nào, chạy offline 100%, độ trễ < 5ms.
"""

import os
import re
import difflib
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# ==============================================================================
# 1. TẬP DỮ LIỆU HUẤN LUYỆN Ý ĐỊNH (INTENT KNOWLEDGE BASE)
# ==============================================================================
INTENT_TRAINING_DATA = {
    "greeting": [
        "xin chào", "chào bạn", "hello", "hi", "chào bot", "chào cinebot",
        "bạn là ai", "chào em", "alo", "helo", "hi cinebot", "chào buổi sáng",
        "chào buổi tối", "có ai ở đây không", "hey bot", "bắt đầu nào", "xin chào trợ lý"
    ],
    "help": [
        "hướng dẫn", "bạn có thể làm gì", "làm sao để dùng", "chỉ tôi cách tìm phim",
        "help", "hướng dẫn sử dụng", "tính năng của bạn", "cách dùng bot",
        "hướng dẫn tôi với", "tôi có thể hỏi gì", "chức năng của bot", "bot làm được gì"
    ],
    "recommend_by_movie": [
        "gợi ý phim giống", "phim tương tự", "phim tựa như", "muốn xem phim giống",
        "phim nào hay như", "phim cùng phong cách", "tương tự phim", "giống phim",
        "gợi ý phim như", "tìm phim giống như", "cho tôi phim tựa tựa", "tôi thích phim",
        "vừa xem xong phim", "có phim nào giống", "đề xuất phim tương tự", "tìm phim tựa như",
        "muốn tìm phim có cảm giác giống", "gợi ý tác phẩm tương tự"
    ],
    "recommend_by_genre": [
        "gợi ý phim hoạt hình", "phim hành động hay", "tìm phim kinh dị", "phim khoa học viễn tưởng",
        "phim hài hước", "phim tình cảm lãng mạn", "phim tài liệu hay", "phim trinh thám hack não",
        "muốn xem phim hoạt hình", "gợi ý phim hành động", "có phim kinh dị nào hay không",
        "tìm phim hài", "muốn xem phim phiêu lưu", "phim chiến tranh lịch sử", "phim tâm lý chính kịch",
        "gợi ý phim sci-fi", "phim hình sự tội phạm"
    ],
    "recommend_by_mood": [
        "hôm nay tôi buồn", "buồn quá muốn xem phim chữa lành", "muốn xem phim hồi hộp giật gân",
        "phim vui vẻ cùng gia đình", "cuối tuần xem gì", "xem phim gì cho đỡ chán",
        "phim cảm động rơi nước mắt", "tôi đang stress cần thư giãn", "muốn xem phim nhẹ nhàng",
        "cần phim giải trí cười sảng khoái", "tối nay xem gì với người yêu", "tâm trạng đang vui"
    ],
    "explain_cosine": [
        "độ tương đồng cosine là gì", "cosine similarity là gì", "tính điểm tương đồng thế nào",
        "giải thích công thức", "tại sao lại gợi ý phim này", "cách tính toán tương đồng",
        "mean centering là gì", "chuẩn hóa vector", "giải thích thuật toán", "công thức toán học",
        "adjusted cosine là gì", "làm sao bạn biết hai phim giống nhau", "nguyên lý gợi ý"
    ],
    "dataset_info": [
        "dữ liệu lấy từ đâu", "tập movielens là gì", "hệ thống có bao nhiêu phim",
        "bao nhiêu người dùng", "thông tin dataset", "nguyên tắc chia train test",
        "data movielens", "có bao nhiêu lượt đánh giá", "nguồn gốc dữ liệu", "dataset gồm những gì"
    ],
    "goodbye": [
        "tạm biệt", "bye bye", "cảm ơn bạn nhé", "hẹn gặp lại", "thank you",
        "thanks", "ok cảm ơn", "tuyệt vời cảm ơn bạn", "cảm ơn cinebot", "tạm biệt nhé"
    ]
}

# Mapping từ khóa thể loại tiếng Việt -> MovieLens Canonical Genres
GENRE_MAP = {
    "hoạt hình": "Animation",
    "anime": "Animation",
    "hài": "Comedy",
    "hài hước": "Comedy",
    "vui nhộn": "Comedy",
    "hành động": "Action",
    "đánh nhau": "Action",
    "kinh dị": "Horror",
    "ma": "Horror",
    "ghê rợn": "Horror",
    "tình cảm": "Romance",
    "lãng mạn": "Romance",
    "yêu": "Romance",
    "viễn tưởng": "Sci-Fi",
    "khoa học viễn tưởng": "Sci-Fi",
    "sci-fi": "Sci-Fi",
    "scifi": "Sci-Fi",
    "chính kịch": "Drama",
    "tâm lý": "Drama",
    "phiêu lưu": "Adventure",
    "thám hiểm": "Adventure",
    "tội phạm": "Crime",
    "hình sự": "Crime",
    "trinh thám": "Mystery",
    "bí ẩn": "Mystery",
    "hack não": "Mystery",
    "chiến tranh": "War",
    "giật gân": "Thriller",
    "hồi hộp": "Thriller",
    "tài liệu": "Documentary",
    "gia đình": "Children",
    "trẻ em": "Children",
    "giả tưởng": "Fantasy",
    "phép thuật": "Fantasy",
    "nhạc kịch": "Musical",
    "ca nhạc": "Musical",
    "cao bồi": "Western",
    "miền tây": "Western"
}

# Mapping tâm trạng -> thể loại phù hợp
MOOD_MAP = {
    "buồn": "Animation|Comedy|Children",
    "chữa lành": "Animation|Comedy|Drama",
    "stress": "Comedy|Animation",
    "thư giãn": "Comedy|Adventure",
    "hồi hộp": "Thriller|Mystery|Action",
    "giật gân": "Thriller|Horror",
    "người yêu": "Romance|Comedy",
    "hẹn hò": "Romance|Comedy",
    "vui vẻ": "Comedy|Children",
    "cảm động": "Drama|Romance"
}


# ==============================================================================
# 2. IN-HOUSE NLP ENGINE CLASS
# ==============================================================================
class CineBotNLPEngine:
    """
    NLP Engine cục bộ của CineBot:
    - Intent Classifier bằng TF-IDF + Cosine Distance
    - Entity Extractor bằng N-gram Fuzzy Match & Regex
    - Response Generator sinh câu trả lời theo ngữ cảnh & gắn thẻ phim
    """

    def __init__(self, movies_df: Optional[pd.DataFrame] = None):
        self.movies_df = movies_df
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
        self.intent_names: List[str] = []
        self.intent_vectors: Optional[np.ndarray] = None
        self._movie_lookup_dict: Dict[str, Tuple[int, str]] = {}
        self._movie_titles_list: List[str] = []

        self._fit_intent_classifier()
        if movies_df is not None:
            self._build_movie_index(movies_df)

    def _fit_intent_classifier(self):
        """Huấn luyện Vector Space Model TF-IDF và tính Centroid Vector cho từng Intent."""
        corpus = []
        labels = []
        for intent, phrases in INTENT_TRAINING_DATA.items():
            for p in phrases:
                corpus.append(p.lower().strip())
                labels.append(intent)

        # Xây dựng TF-IDF Vectorizer
        tfidf_matrix = self.vectorizer.fit_transform(corpus)

        # Tính Centroid Vector cho từng intent
        self.intent_names = sorted(list(INTENT_TRAINING_DATA.keys()))
        centroids = []
        labels_arr = np.array(labels)
        
        for intent in self.intent_names:
            idx = np.where(labels_arr == intent)[0]
            intent_vecs = tfidf_matrix[idx]
            # Vector trọng tâm (Centroid Vector) của intent
            centroid = intent_vecs.mean(axis=0)
            centroids.append(np.asarray(centroid).flatten())

        # Chuẩn hóa L2 các centroid để tính Cosine Similarity nhanh chóng
        centroids = np.array(centroids)
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.intent_vectors = centroids / norms
        logger.info(f"CineBot NLP: Đã nạp {len(self.intent_names)} intents và {len(corpus)} mẫu hội thoại.")

    def _build_movie_index(self, movies_df: pd.DataFrame):
        """Xây dựng chỉ mục tên phim để tìm kiếm gần đúng (Fuzzy Match)."""
        self._movie_lookup_dict.clear()
        self._movie_titles_list.clear()

        for _, row in movies_df.iterrows():
            m_id = int(row["movieId"])
            title = str(row["title"])
            self._movie_titles_list.append(title)

            # Chuẩn hóa: "Toy Story (1995)" -> "toy story"
            raw_clean = re.sub(r"\(\d{4}\)", "", title).strip()
            clean_title = raw_clean.lower()
            if clean_title:
                self._movie_lookup_dict[clean_title] = (m_id, title)
            self._movie_lookup_dict[title.lower()] = (m_id, title)

            # Tháo đảo ngữ mạo từ MovieLens: "Matrix, The" -> "The Matrix" và "Matrix"
            for article in [", The", ", A", ", An"]:
                if raw_clean.endswith(article):
                    prefix = article.replace(", ", "") + " "
                    base_name = raw_clean[:-len(article)].strip()
                    natural_title = (prefix + base_name).lower()
                    self._movie_lookup_dict[natural_title] = (m_id, title)
                    if len(base_name) >= 3:
                        self._movie_lookup_dict[base_name.lower()] = (m_id, title)

        # Danh sách các title chuẩn hóa sắp xếp theo độ dài giảm dần để ưu tiên match cụm dài trước
        self._sorted_clean_keys = sorted(
            [k for k in self._movie_lookup_dict.keys() if len(k) >= 3 and not k.endswith(")")],
            key=len,
            reverse=True
        )

        logger.info(f"CineBot NLP: Đã tạo chỉ mục tìm kiếm cho {len(self._movie_titles_list)} bộ phim ({len(self._movie_lookup_dict)} khóa ánh xạ).")

    def set_movies_df(self, movies_df: pd.DataFrame):
        """Cập nhật dữ liệu phim khi server nạp dữ liệu xong."""
        self.movies_df = movies_df
        self._build_movie_index(movies_df)

    # --------------------------------------------------------------------------
    # BƯỚC 1: CLASSIFY INTENT DÙNG TF-IDF & COSINE SIMILARITY
    # --------------------------------------------------------------------------
    def classify_intent(self, text: str) -> Tuple[str, float]:
        """
        Phân loại ý định của người dùng bằng cách tính Cosine Similarity 
        giữa vector TF-IDF của câu hỏi với Centroid Vector của các Intent.
        """
        text_clean = text.lower().strip()
        vec = self.vectorizer.transform([text_clean])
        
        # Nếu câu hỏi quá ngắn hoặc toàn từ ngoài từ vựng
        if vec.nnz == 0:
            movie_match = self.extract_movie_entity(text)
            if movie_match:
                return "recommend_by_movie", 0.90
            return "fallback", 0.0

        vec_dense = vec.toarray().flatten()
        norm = np.linalg.norm(vec_dense)
        if norm > 0:
            vec_dense = vec_dense / norm

        # Tính Cosine Similarity: S = vec_dense @ intent_vectors.T
        scores = self.intent_vectors @ vec_dense
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        best_intent = self.intent_names[best_idx]

        # Kiểm tra xem có chứa tên phim rõ ràng không (ưu tiên cao)
        movie_match = self.extract_movie_entity(text)
        if movie_match and best_intent in ["recommend_by_movie", "fallback", "greeting"]:
            return "recommend_by_movie", max(best_score, 0.85)

        # Kiểm tra ngưỡng tin cậy
        if best_score < 0.18:
            if movie_match:
                return "recommend_by_movie", 0.85
            return "fallback", best_score

        return best_intent, best_score

    # --------------------------------------------------------------------------
    # BƯỚC 2: TRÍCH XUẤT THỰC THỂ (PHIM, THỂ LOẠI, TÂM TRẠNG)
    # --------------------------------------------------------------------------
    def extract_movie_entity(self, text: str) -> Optional[Tuple[int, str, float]]:
        """
        Trích xuất tên phim từ câu chat:
        1. Quét tên phim xuất hiện trực tiếp trong câu theo ranh giới từ.
        2. Quét cụm từ chỉ định: "giống [phim]", "tương tự [phim]", "như [phim]"
        3. Dùng difflib trên các cụm trích xuất.
        Trả về: (movieId, official_title, confidence) hoặc None.
        """
        if not self._movie_lookup_dict:
            return None

        text_lower = text.lower().strip()
        padded_text = f" {text_lower} "

        # 1. Quét trực tiếp các tên phim nổi bật có trong câu
        for clean_key in self._sorted_clean_keys:
            # Chỉ xét các tên phim dài >= 3 ký tự và không phải từ tiếng Việt thông dụng
            if clean_key in ["the", "and", "phim", "nào", "hay", "cho", "với", "nhé", "xem"]:
                continue
            
            # Kiểm tra xem clean_key có xuất hiện nguyên vẹn trong câu không
            if f" {clean_key} " in padded_text or f"'{clean_key}'" in padded_text or f'"{clean_key}"' in padded_text:
                m_id, full_title = self._movie_lookup_dict[clean_key]
                return m_id, full_title, 1.0

        # 2. Tìm theo cụm từ mẫu nhận diện
        patterns = [
            r"(?:giống|như|tương tự|tựa như|phim)\s+[\"']?([^\"',.?!]+)[\"']?",
            r"xem\s+[\"']?([^\"',.?!]+)[\"']?",
            r"thích\s+[\"']?([^\"',.?!]+)[\"']?"
        ]

        candidates = []
        for pat in patterns:
            matches = re.findall(pat, text_lower)
            for m in matches:
                cand = m.strip()
                # Loại bỏ các từ phụ trợ tiếng Việt
                for stop in ["với", "nhé", "ạ", "đi", "nào", "hay", "không", "giúp tôi", "giúp mình", "xem sao", "cho tôi"]:
                    cand = re.sub(r'\b' + re.escape(stop) + r'\b', '', cand).strip()
                if len(cand) >= 2 and cand not in ["phim", "gì", "nào", "hay", "tôi", "bạn"]:
                    candidates.append(cand)

        # 3. Thử matching candidates với database
        for cand in candidates:
            if cand in self._movie_lookup_dict:
                m_id, full_title = self._movie_lookup_dict[cand]
                return m_id, full_title, 0.95

            # Fuzzy match qua difflib
            close = difflib.get_close_matches(cand, self._sorted_clean_keys, n=1, cutoff=0.72)
            if close:
                matched_key = close[0]
                m_id, full_title = self._movie_lookup_dict[matched_key]
                return m_id, full_title, 0.85

        return None

    def extract_genre_entity(self, text: str) -> Optional[str]:
        """Trích xuất thể loại phim từ câu hỏi."""
        text_lower = text.lower()
        for kw, genre in GENRE_MAP.items():
            # Tìm từ nguyên vẹn
            if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
                return genre
        return None

    def extract_mood_entity(self, text: str) -> Optional[str]:
        """Trích xuất tâm trạng người dùng."""
        text_lower = text.lower()
        for mood_kw, genre_filter in MOOD_MAP.items():
            if mood_kw in text_lower:
                return genre_filter
        return None

    # --------------------------------------------------------------------------
    # BƯỚC 3: DYNAMIC RESPONSE GENERATION (SINH PHẢN HỒI THEO NGỮ CẢNH)
    # --------------------------------------------------------------------------
    def process_message(
        self, 
        message: str, 
        recommender: Any, 
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Xử lý toàn diện một thông điệp từ người dùng:
        - Phân loại ý định
        - Trích xuất thực thể
        - Gọi thuật toán CosineItemItemRecommender
        - Trả về câu trả lời Markdown + Thẻ phim tương tác
        """
        intent, confidence = self.classify_intent(message)
        logger.info(f"CineBot NLU: message='{message}' -> intent='{intent}' (conf={confidence:.2f})")

        genre_entity = self.extract_genre_entity(message)
        movie_entity = self.extract_movie_entity(message)
        mood_entity = self.extract_mood_entity(message)

        # Xử lý theo từng Intent:
        if intent == "greeting":
            return {
                "intent": intent,
                "reply": (
                    "👋 **Chào bạn! Tôi là CineBot — Trợ lý điện ảnh ảo của hệ thống CineVector.**\n\n"
                    "Tôi được trang bị bộ xử lý ngôn ngữ tự nhiên **In-House NLP** (sử dụng TF-IDF & Cosine Similarity) "
                    "và kết nối trực tiếp với ma trận 610 chiều người dùng MovieLens.\n\n"
                    "💡 **Bạn có thể thử trò chuyện với tôi:**\n"
                    "- *'Gợi ý cho tôi phim giống Toy Story'* hoặc *'Tìm phim như Inception'*\n"
                    "- *'Có phim hoạt hình nào hay không?'*\n"
                    "- *'Hôm nay tôi buồn, gợi ý phim chữa lành nhẹ nhàng'*\n"
                    "- *'Độ tương đồng Cosine được tính như thế nào?'*"
                ),
                "recommendations": []
            }

        elif intent == "help":
            return {
                "intent": intent,
                "reply": (
                    "🛠️ **Hướng dẫn sử dụng CineBot:**\n\n"
                    "1. **Tìm phim tương tự:** Nhập câu hỏi kèm tên phim yêu thích (VD: *'Tìm phim giống The Matrix'*, *'Phim nào hay như Titanic'*).\n"
                    "2. **Khám phá theo thể loại:** Nhập thể loại bạn muốn xem (VD: *'Gợi ý phim trinh thám hack não'*, *'Phim khoa học viễn tưởng'*).\n"
                    "3. **Tư vấn theo tâm trạng:** Nói cho tôi biết bạn đang cảm thấy thế nào (VD: *'Đang stress cần phim hài cười xả láng'*, *'Buồn cần phim chữa lành'*).\n"
                    "4. **Hỏi về giải thuật:** Gõ *'Giải thích công thức Cosine'* để xem nguyên lý toán học đằng sau hệ thống."
                ),
                "recommendations": []
            }

        elif intent == "explain_cosine":
            return {
                "intent": intent,
                "reply": (
                    "📐 **Nguyên lý toán học của Hệ thống Gợi ý (Adjusted Cosine Similarity):**\n\n"
                    "1. **Mô hình hóa Vector Item–User:**\n"
                    "Mỗi bộ phim $i$ được biểu diễn bằng một vector $\\vec{u}_i \\in \\mathbb{R}^{610}$, "
                    "trong đó mỗi chiều đại diện cho đánh giá của một người dùng trong tập Train.\n\n"
                    "2. **Khử thiên vị (Mean-Centering / Adjusted Cosine):**\n"
                    "Để tránh trường hợp người dùng 'dễ dãi' (toàn chấm 5 sao) hoặc 'khắt khe' (toàn chấm 1-2 sao), "
                    "hệ thống trừ đi điểm trung bình của mỗi user: $r'_{u, i} = r_{u, i} - \\bar{r}_u$.\n\n"
                    "3. **Độ tương đồng Cosine giữa 2 phim $i$ và $j$:**\n"
                    "$$\\text{Cosine}(i, j) = \\frac{\\vec{u}_i \\cdot \\vec{u}_j}{\\|\\vec{u}_i\\|_2 \\|\\vec{u}_j\\|_2}$$\n\n"
                    "Điểm Cosine nằm trong đoạn $[0, 1]$. Điểm càng gần 1 chứng tỏ hai bộ phim có thị hiếu khán giả đánh giá cực kỳ tương đồng!"
                ),
                "recommendations": []
            }

        elif intent == "dataset_info":
            return {
                "intent": intent,
                "reply": (
                    "📊 **Thông tin tập dữ liệu MovieLens (ml-latest-small):**\n\n"
                    "- **Quy mô:** 100.836 đánh giá từ **610 người dùng** trên **9.742 bộ phim**.\n"
                    "- **Độ thưa của ma trận tương tác:** ~98.3% (được nén hiệu quả dưới dạng CSR Sparse Matrix).\n"
                    "- **Nguyên tắc phân chia (Zero Data Leakage):**\n"
                    "  + **Train (70%):** Dùng để dựng ma trận vector và tính độ tương đồng Cosine.\n"
                    "  + **Validation (10%) & Test (20%):** Được đóng băng độc lập theo thời gian (Stratified Temporal Split) "
                    "để đo lường các metric khách quan (`HitRate@10 = 31.48%`, `Precision@10 = 7.70%`)."
                ),
                "recommendations": []
            }

        elif intent == "goodbye":
            return {
                "intent": intent,
                "reply": (
                    "👋 **Tạm biệt bạn!** Chúc bạn có những giờ phút xem phim thật tuyệt vời và thư giãn! "
                    "Khi nào cần gợi ý phim mới, CineBot luôn sẵn sàng hỗ trợ nhé! 🎬🍿"
                ),
                "recommendations": []
            }

        elif intent == "recommend_by_movie" or movie_entity is not None:
            # Trường hợp tìm phim tương tự theo movie
            if movie_entity is not None:
                m_id, m_title, conf = movie_entity
                if recommender is not None:
                    res = recommender.recommend(movie_id=m_id, top_k=top_k, filter_genre=genre_entity)
                    recs = res.get("recommendations", [])
                    status = res.get("status", "success")
                    warning = res.get("warning")

                    if status == "warning" or not recs:
                        return {
                            "intent": "recommend_by_movie",
                            "referenced_movie": {"movieId": m_id, "title": m_title},
                            "reply": (
                                f"⚠️ Bộ phim **{m_title}** có quá ít lượt đánh giá trong tập dữ liệu Train "
                                f"(ngưỡng tối thiểu là {recommender.min_movie_ratings} lượt). "
                                f"Hệ thống tuân thủ nguyên tắc chống suy biến nên không thể tính toán vector tương đồng tin cậy.\n\n"
                                f"💡 Bạn hãy thử các tựa phim phổ biến hơn như: *Toy Story, Inception, Titanic, The Dark Knight, Forrest Gump* nhé!"
                            ),
                            "recommendations": []
                        }

                    intro = (
                        f"🎯 **Tuyệt vời! Nếu bạn đã mê phim *{m_title}*:**\n\n"
                        f"Hệ thống đã phân tích không gian vector tương tác của 610 người dùng trên MovieLens "
                        f"và tìm thấy **Top {len(recs)} bộ phim tương đồng nhất**:"
                    )
                    if warning:
                        intro += f"\n\n> ⚠️ *{warning}*"

                    return {
                        "intent": "recommend_by_movie",
                        "referenced_movie": {"movieId": m_id, "title": m_title},
                        "reply": intro,
                        "recommendations": recs
                    }

            # Không trích xuất được tên phim rõ ràng
            return {
                "intent": "recommend_by_movie",
                "reply": (
                    "🤔 Bạn muốn tìm phim tương tự một bộ phim nào đó, nhưng tôi chưa nhận diện được chính xác tên phim trong câu của bạn.\n\n"
                    "💡 **Mẹo:** Bạn hãy gõ tên phim rõ hơn nhé, ví dụ:\n"
                    "- *'Gợi ý phim giống Toy Story'*\n"
                    "- *'Tìm phim tựa như The Matrix'*\n"
                    "- *'Phim nào hay như Inception'*"
                ),
                "recommendations": []
            }

        elif intent == "recommend_by_genre" or genre_entity is not None:
            # Tìm phim theo thể loại
            target_genre = genre_entity or "Animation"
            top_movies_in_genre = self._get_popular_movies_by_genre(recommender, target_genre, limit=top_k)
            
            return {
                "intent": "recommend_by_genre",
                "target_genre": target_genre,
                "reply": (
                    f"🎬 **Dưới đây là Top {len(top_movies_in_genre)} bộ phim thể loại *{target_genre}* "
                    f"được cộng đồng người dùng MovieLens đánh giá cao nhất:**\n\n"
                    f"Bạn có thể chọn một phim bất kỳ bên dưới để tôi tìm tiếp các bộ phim có phong cách tương đồng nhé!"
                ),
                "recommendations": top_movies_in_genre
            }

        elif intent == "recommend_by_mood" or mood_entity is not None:
            # Tư vấn theo tâm trạng
            genre_filter = mood_entity or "Animation|Comedy"
            primary_genre = genre_filter.split("|")[0]
            top_movies_in_mood = self._get_popular_movies_by_genre(recommender, primary_genre, limit=top_k)

            return {
                "intent": "recommend_by_mood",
                "reply": (
                    f"🌈 **Tôi hiểu tâm trạng của bạn rồi!** Một bộ phim phù hợp sẽ là liều thuốc tinh thần tuyệt vời.\n\n"
                    f"Dựa trên phân tích thể loại (*{genre_filter.replace('|', ', ')}*), "
                    f"tôi gợi ý cho bạn những bộ phim được khán giả đánh giá rất tích cực:"
                ),
                "recommendations": top_movies_in_mood
            }

        else:
            # Fallback
            return {
                "intent": "fallback",
                "reply": (
                    "🤖 Xin lỗi, tôi chưa hiểu rõ yêu cầu này của bạn lắm!\n\n"
                    "Tôi là trợ lý chuyên sâu về **gợi ý phim theo độ tương đồng Cosine**. "
                    "Bạn có thể thử bấm vào một trong các câu hỏi gợi ý bên dưới hoặc hỏi tôi về một bộ phim cụ thể nhé! 👇"
                ),
                "recommendations": []
            }

    def _get_popular_movies_by_genre(
        self, 
        recommender: Any, 
        genre: str, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Lấy danh sách phim tiêu biểu theo thể loại từ tập dữ liệu đã học."""
        if self.movies_df is None or recommender is None:
            return []

        matched = self.movies_df[self.movies_df["genres"].str.contains(genre, case=False, na=False)]
        results = []

        for _, row in matched.iterrows():
            m_id = int(row["movieId"])
            stats = recommender.movie_stats.get(m_id, {})
            count = stats.get("rating_count", 0)
            mean = stats.get("rating_mean", 0.0)

            # Lọc những phim có số lượng rating đáng tin cậy (>= 20 ratings)
            if count >= 20 and mean >= 3.5:
                results.append({
                    "movieId": m_id,
                    "title": row["title"],
                    "genres": row["genres"],
                    "similarity_score": 1.0,
                    "rating_count": count,
                    "rating_mean": round(mean, 2),
                    "reason": f"Phim tiêu biểu thể loại {genre} với {count} lượt đánh giá (Rating: {mean:.1f}/5.0)"
                })

        # Sắp xếp theo weighted rating
        results.sort(key=lambda x: (x["rating_mean"], x["rating_count"]), reverse=True)
        return results[:limit]
