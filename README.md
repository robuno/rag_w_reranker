# RAG Evaluator with Rerank

This project aims to create a RAG pipeline by combining HuggingFace models with LangChain Cohere Rerank modules and create a ready notebook to evaluate the results. 

Apart from this, a script has been added to download the pages in a documentation and parse the HTML codes with BeautifulSoup4.

These code snippets were combined by following the instructions in the HuggingFace [[1]](https://huggingface.co/learn/cookbook/advanced_rag), LangChain [[2](https://python.langchain.com/docs/integrations/retrievers/cohere-reranker), [3](https://python.langchain.com/docs/expression_language/cookbook/prompt_llm_parser)], Ragas [[4]](https://docs.ragas.io/en/latest/howtos/applications/compare_llms.html) documentation. 



The project includes:
- RAG pipeline that allows trying different HuggingFace models
- Evaluating the results of hyperparameter search & RAG pipeline with RAGAS
- FAISS vector database
- HuggingFacePipeline with prompt template
- a script which allows to download the pages in a documentation and parse the HTML codes with BeautifulSoup4
- Chunking Q&A datasets CSV format files and evaluating them through RAG

External libraries and packages:
- Torch
- LangChain
- FAISS
- Ragas
- HuggingFace
- Sentence-transformers
- BitsAndBytes & Accelerate
- PyPDF
- Cohere
- Pandas, Matplotlib, tqdm, getpass


