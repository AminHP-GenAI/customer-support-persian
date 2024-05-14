from typing import List, Dict, Any, Iterator, Type, Optional

import os
import json
import re
import bs4

from langchain_core.documents import Document
from langchain_text_splitters.character import TextSplitter, _split_text_with_regex
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.web_base import WebBaseLoader, _build_metadata
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel

from llm_translation import translate_to_persian


class FaqWebBaseLoader(WebBaseLoader):

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load text from the url(s) in web_path."""
        for path in self.web_paths:
            soup = self._scrape(path, bs_kwargs=self.bs_kwargs)
            for subject, faq_texts in self._extract_faqs(soup).items():
                metadata = _build_metadata(soup, path)
                metadata['subject'] = subject
                text = "\n\n".join(faq_texts)
                yield Document(page_content=text, metadata=metadata)

    def _extract_faqs(self, soup: bs4.BeautifulSoup) -> Dict[str, List[str]]:
        faqs_tag = soup.find('body').find('script')
        faqs_json = json.loads(
            faqs_tag.contents[0].replace('window.__SSR_CONTEXT__ = ', '')
        )['cms_hc-products']

        result = {}

        for subject_faqs_json in faqs_json:
            subject_name = subject_faqs_json['title']
            result[subject_name] = []

            for faq_info in subject_faqs_json['faq']:
                question = faq_info['question']
                answer = faq_info['answer']
                faq_text = f"FAQ_QUESTION:\n{question}\nFAQ_ANSWER:\n{answer}"
                result[subject_name].append(faq_text)

        return result


class FaqTextSplitter(TextSplitter):

    def __init__(
        self, separator: str = "\n\n", is_separator_regex: bool = False, **kwargs: Any
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(**kwargs)
        self._separator = separator
        self._is_separator_regex = is_separator_regex

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        # First we naively split the large input into a bunch of smaller ones.
        separator = (
            self._separator if self._is_separator_regex else re.escape(self._separator)
        )
        splits = _split_text_with_regex(text, separator, self._keep_separator)
        return splits


class Policy:

    def __init__(
        self,
        data_dir: str,
        llm: BaseChatModel,
        embedding: Embeddings,
        k: int = 5
    ) -> None:
        self.data_dir = data_dir
        self.llm = llm
        self.embedding = embedding
        self.vectorstore = self.get_or_create_vectorstore()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={'k': k})

    @property
    def vectorstore_path(self) -> str:
        return os.path.join(self.data_dir, 'chroma.sqlite3')

    def download_faqs(self) -> Iterator[Document]:
        loader = FaqWebBaseLoader(
            web_paths=["https://www.alibaba.ir/help-center/categories/faq"]
        )
        documents = loader.load()
        return documents

    def split_documents_qa(self, documents: Iterator[Document]) -> Iterator[Document]:
        text_splitter = FaqTextSplitter()
        splits = text_splitter.split_documents(documents)
        return splits

    def split_documents_chunk1000(self, documents: Iterator[Document]) -> Iterator[Document]:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        return splits

    def create_vectorstore(self, documents: Iterator[Document]) -> Chroma:
        return Chroma.from_documents(
            documents=documents, 
            embedding=self.embedding,
            persist_directory=self.data_dir,
        )

    def load_vectorstore(self) -> Chroma:
        if not os.path.exists(self.vectorstore_path):
            raise FileNotFoundError(f"vectorstore file not found '{self.vectorstore_path}'")

        return Chroma(
            embedding_function=self.embedding,
            persist_directory=self.data_dir,
        )

    def get_or_create_vectorstore(self) -> Chroma:
        try:
            return self.load_vectorstore()
        except FileNotFoundError:
            pass

        documents = self.download_faqs()
        splits_qa = self.split_documents_qa(documents)
        return self.create_vectorstore(splits_qa)

    def get_relevant_documents(self, query: str) -> Iterator[Document]:
        return self.retriever.invoke(query)

    def get_tools(self) -> Dict[str, BaseTool]:
        tools = [
            LookupPolicyTool(policy=self),
        ]
        return {tool.name: tool for tool in tools}



class LookupPolicyToolInput(BaseModel):
    query: str = Field(description="should be a search query")


class LookupPolicyTool(BaseTool):

    name = "lookup_policy_tool"
    description = (
        "Consult the company policies to check whether certain options are permitted.\n"
        "Use this before answering the general questions about the company policies.\n"
        "Use this before making any flight changes performing other 'write' events."
    )
    args_schema: Type[BaseModel] = LookupPolicyToolInput
    return_direct: bool = False

    policy: Policy

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        docs = self.policy.get_relevant_documents(translate_to_persian(query, self.policy.llm))
        return '\n\n'.join([
            f"FAQ_SUBJECT: {doc.metadata['subject']}\n{doc.page_content}"
            for doc in docs
        ])
