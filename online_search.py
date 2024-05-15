from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_community.tools.tavily_search import TavilyAnswer
from langchain_core.callbacks import CallbackManagerForToolRun

from llm_translation import translate_to_persian


class PersianTavilySearchTool(TavilyAnswer):

    name: str = 'tavily_search_tool'
    description: str = (
        "A search engine optimized for comprehensive, accurate, and trusted results. "
        "Useful for when you need to answer questions about current events like weather conditions, etc. "
        "Input should be a search query."
    )
    max_results: int = 20

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
            result_text = self.api_wrapper.raw_results(
                persian_query,
                max_results=self.max_results,
                include_answer=True,
                search_depth='basic',
            )['answer']

            return result_text
        except Exception as e:
            return repr(e)
