from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.callbacks import CallbackManagerForToolRun

from llm_translation import translate_to_english, translate_to_persian


class PersianTavilySearchResults(TavilySearchResults):

    llm: BaseChatModel

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        try:
            # Translate the query to Persian
            persian_query = translate_to_persian(query, self.llm)

            # Search the Persian query using Tavily API
            result_list = self.api_wrapper.results(
                persian_query,
                self.max_results,
            )

            # Translate the result back to English
            # Results are merged into a single text to reduce the number of LLM translation calls
            result_text = '\n\n'.join([item['content'] for item in result_list])
            english_result = translate_to_english(result_text, self.llm)

            return english_result
        except Exception as e:
            return repr(e)
