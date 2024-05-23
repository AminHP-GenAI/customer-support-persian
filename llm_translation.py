from langchain_core.language_models import BaseChatModel


def translate_to_english(text: str, llm: BaseChatModel) -> str:
    raw_result = llm.invoke(
        "Translate the following Persian text to English (put the translated text inside {}):\n\n" + text
    ).content
    result = raw_result[raw_result.index('{')+1:raw_result.rindex('}')]
    return result


def translate_to_persian(text: str, llm: BaseChatModel) -> str:
    result = llm.invoke(
        "Translate the following text to Persian without adding any notes.\n" +
        "Do not convert dates to Jalali calander and keep times in GMT.\n" +
        "Do not say anything before or after the translated text.\n" +
        "Return the text without any changes if it was already in Persian.\n"
        "Here is the text to translate:\n\n" +
        text
    ).content
    return result
