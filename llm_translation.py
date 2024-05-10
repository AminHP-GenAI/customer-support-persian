from langchain_core.language_models import BaseChatModel


def translate_to_english(text: str, llm: BaseChatModel) -> str:
    return llm.invoke(
        "Translate the following text to English without adding any notes:\n\n" + text
    ).content


def translate_to_persian(text: str, llm: BaseChatModel) -> str:
    return llm.invoke(
        "Translate the following text to Persian without adding any notes.\n" +
        "Do not convert dates to Jalali calander and keep times in GMT.\n" +
        "Do not say anything before or after the translated text.\n" +
        "Here is the text to translate:\n\n" +
        text
    ).content
