import pandas as pd
import pytest
from src.chatbot_engine import CineBotNLPEngine


@pytest.fixture(scope="module")
def bot():
    movies = pd.DataFrame({"movieId": [1, 2], "title": ["Toy Story (1995)", "Jumanji (1995)"], "genres": ["Animation|Comedy", "Adventure"]})
    return CineBotNLPEngine(movies)


@pytest.mark.parametrize("message,intent", [("xin chào", "greeting"), ("độ tương đồng cosine là gì", "explain_cosine"), ("tập dữ liệu movielens gồm những gì", "dataset_info")])
def test_bot_information_intents(bot, message, intent):
    response = bot.process_message(message, None, top_k=3)
    assert response["intent"] == intent
    assert response["reply"]


def test_movie_entity_is_resolved_from_catalog(bot):
    entity = bot.extract_movie_entity("gợi ý phim giống Toy Story")
    assert entity is not None and entity[0] == 1
