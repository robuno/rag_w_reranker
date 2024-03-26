from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.document_compressors import CohereRerank
from langchain.retrievers import  ContextualCompressionRetriever
from ragas import evaluate
import cohere
from cohere import Client
from datasets import Dataset
# from langchain.retrievers import CohereRerank, ContextualCompressionRetriever
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch

PROMPT_TEMP = """
<|system|>
You question-answering task assistant.
Answer the question only based on your context and knowledge.
If you do not know the answer with given context, say that you do not know.
Use the following context:

{context}

</s>
<|user|>
{question}
</s>
<|assistant|>

"""


def get_chunked_docs(CHUNK_SIZE, docs):

    splitter = RecursiveCharacterTextSplitter(chunk_size = CHUNK_SIZE,
                                            chunk_overlap = 64,
                                            add_start_index=True,             # add chunk's start index in metadata
                                            #   strip_whitespace=True,        # start and end of every doc
                                            )

    chunked_docs = splitter.split_documents(docs)
    # print(len(chunked_docs))

    return chunked_docs



def get_emb_model(model_name, device):
    device = "cuda"

    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': True}  

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        multi_process=True,
        model_kwargs = model_kwargs,
        encode_kwargs = encode_kwargs
    )


def get_db_retriever(chunked_docs, embeddings, SEARCH_KWARG):
    db = FAISS.from_documents(
                chunked_docs,
                embeddings,
                distance_strategy = DistanceStrategy.COSINE
                )

    retriever = db.as_retriever(
                # search_type ="mmr",               # "similarity"
                search_kwargs={"k": SEARCH_KWARG},
                )
    
    return db, retriever



def get_model_tokenizer(model_name):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_use_double_quant = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                quantization_config=bnb_config
                                                )

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                            # padding=True,
                                            # truncation=True
                                            )

    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def create_hf_pipeline(model, tokenizer, model_temperature, max_new_token= 1024):

    text_generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=model_temperature,
        do_sample=True,
        repetition_penalty=1.1,
        return_full_text=True,
        max_new_tokens=max_new_token,
        batch_size=2,
    )

    text_generation_pipeline.model.config.pad_token_id = text_generation_pipeline.model.config.eos_token_id

    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    return llm 

def create_comp_retriever(cohere_api_key, retriever):
    class CustomCohereRerank(CohereRerank):
        class Config(CohereRerank.Config):
            arbitrary_types_allowed = True

    CustomCohereRerank.update_forward_refs()
    compressor = CustomCohereRerank(cohere_api_key=cohere_api_key,
                                    client=Client,
                                    top_n=4)

    compressor = CohereRerank()
    comp_retriever = ContextualCompressionRetriever(
                                base_compressor = compressor,
                                base_retriever = retriever
                            )
    
    return comp_retriever


def create_qa_chain(llm, retriever, active_doc_source = True):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,       # retriever or compression_retriever
        # chain_type="refine",
        return_source_documents=active_doc_source,
        )
    
    

def create_llm_chain_with_prompt(llm, prompt_template = PROMPT_TEMP):

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TEMP,
    )

    llm_chain = prompt | llm | StrOutputParser()

    # llm_chain = (
    #     {"context": compression_retriever,  "question": RunnablePassthrough()}
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    # )

    return llm_chain

def evaluate_answers(questions_dset, 
                     ground_truths_dset, 
                     rag_chain, 
                     comp_retriever, 
                     llm, 
                     emb_model, 
                     ragas_metrics):
    answers_rag = []
    contexts_rag = []
    for query in questions_dset:
        answers_rag.append(rag_chain.invoke(query))
        contexts_rag.append([d.page_content for d in comp_retriever.get_relevant_documents(query)])

    data_test_dict = {
        "question": questions_dset,
        "answer": answers_rag,
        "contexts": contexts_rag,
        "ground_truths": ground_truths_dset
    }
    test_dset = Dataset.from_dict(data_test_dict)


    result_testset = evaluate(
        test_dset,
        llm = llm,
        embeddings = emb_model,
        metrics=ragas_metrics,
    )

    return result_testset