"""Module for querying indexed LinkedIn profile data."""

import logging
from typing import Any

from llama_index.core import PromptTemplate, VectorStoreIndex

import config
from modules.llm_interface import create_openai_llm

logger = logging.getLogger(__name__)
EMPTY_MARKERS = {"", "none", "null", "empty response"}


def generate_initial_facts(index: VectorStoreIndex) -> str:
    """Generates interesting facts about the person's career or education.

    Args:
        index: VectorStoreIndex containing the LinkedIn profile data.

    Returns:
        String containing interesting facts about the person.
    """
    try:
        # Create LLM for generating facts
        openai_llm = create_openai_llm(temperature=0.0, max_new_tokens=500, decoding_method="sample")

        # Create prompt template
        facts_prompt = PromptTemplate(template=config.INITIAL_FACTS_TEMPLATE)

        # Create query engine
        query_engine = index.as_query_engine(
            streaming=False,
            similarity_top_k=config.SIMILARITY_TOP_K,
            llm=openai_llm,
            text_qa_template=facts_prompt,
        )

        # Execute the query
        query = "Provide three interesting facts about this person's career or education."
        response = query_engine.query(query)
        response_text = getattr(response, "response", None)
        if (
            isinstance(response_text, str)
            and response_text.strip()
            and response_text.strip().lower() not in EMPTY_MARKERS
        ):
            return response_text.strip()

        fallback_text = str(response).strip()
        if fallback_text and fallback_text.lower() not in EMPTY_MARKERS:
            return fallback_text

        logger.warning("generate_initial_facts returned an empty response.")
        return "I couldn't generate initial facts from the indexed profile data."
    except Exception as e:
        logger.error(f"Error in generate_initial_facts: {e}")
        return "Failed to generate initial facts."


def answer_user_query(index: VectorStoreIndex, user_query: str) -> Any:
    """Answers the user's question using the vector database and the LLM.

    Args:
        index: VectorStoreIndex containing the LinkedIn profile data.
        user_query: The user's question.

    Returns:
        Response object containing the answer to the user's question.
    """
    try:
        # Create prompt template
        question_prompt = PromptTemplate(template=config.USER_QUESTION_TEMPLATE)

        models_to_try = [config.LLM_MODEL_ID, "gpt-4o-mini", "gpt-4.1-mini"]
        tried_models = set()

        for model_id in models_to_try:
            if model_id in tried_models:
                continue
            tried_models.add(model_id)

            llm = create_openai_llm(
                temperature=0.0,
                max_new_tokens=config.MAX_NEW_TOKENS,
                decoding_method="greedy",
                model_id=model_id,
            )

            query_engine = index.as_query_engine(
                streaming=False,
                similarity_top_k=config.SIMILARITY_TOP_K,
                llm=llm,
                text_qa_template=question_prompt,
            )

            answer = query_engine.query(user_query)
            answer_text = getattr(answer, "response", None)
            if (
                isinstance(answer_text, str)
                and answer_text.strip()
                and answer_text.strip().lower() not in EMPTY_MARKERS
            ):
                return answer_text.strip()

            fallback_text = str(answer).strip()
            if fallback_text and fallback_text.lower() not in EMPTY_MARKERS:
                return fallback_text

            logger.warning("Model %s returned empty output", model_id)

        logger.warning("answer_user_query returned an empty response.")
        return "I couldn't find an answer in the indexed profile data. Try asking a more specific question."
    except Exception as e:
        logger.error(f"Error in answer_user_query: {e}")
        return "Failed to get an answer."
