import os
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Tuple
from llama_index.embeddings import OpenAIEmbedding
from llama_index.indices.postprocessor import LLMRerank
from llama_index.llms import OpenAI as OpenAILLM, ChatMessage
from llama_index.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.schema import BaseNode
from llama_index import ServiceContext, VectorStoreIndex, download_loader, StorageContext
from messages_dictionary import messages_dict
from sys import argv
import json
from PDFReader import PDFReader
from docxReader import DocxReader
import re
import streamlit as st

# Ensure OPENAI_API_KEY is set in the environment
api_key = os.getenv('OPENAI_API_KEY')
if api_key is None:
    raise EnvironmentError("OPENAI_API_KEY not found in environment variables. Please set it before running the script.")

# Initialize embedding model and LLM
embed_model = OpenAIEmbedding(model="text-embedding-3-large")
llm = OpenAILLM(api_key=api_key, model="gpt-4-0125-preview", temperature=0, top_p=0.1)

def process_documents(file_path: Path) -> List[BaseNode]:
    if file_path.suffix == ".docx":
        loader = DocxReader()
    elif file_path.suffix == ".pdf":
        loader = PDFReader()
    documents = loader.load_data(file=file_path)
    node_parser_instance = HierarchicalNodeParser.from_defaults(chunk_sizes=[512, 256, 128])
    nodes = node_parser_instance.get_nodes_from_documents(documents)
    return get_leaf_nodes(nodes)

def create_service_and_storage_contexts(llm: OpenAILLM, embed_model: OpenAIEmbedding, nodes: List[BaseNode]) -> Tuple[ServiceContext, StorageContext, VectorStoreIndex]:
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)
    vector_store_index = VectorStoreIndex(nodes, storage_context=storage_context, service_context=service_context)
    return service_context, storage_context, vector_store_index

def setup_query_engine(service_context: ServiceContext, vector_store_index: VectorStoreIndex) -> RetrieverQueryEngine:
    automerging_retriever = vector_store_index.as_retriever(similarity_top_k=8)
    rerank_processor = LLMRerank(top_n=4)
    query_engine = RetrieverQueryEngine.from_args(retriever=automerging_retriever, node_postprocessors=[rerank_processor], verbose=True, service_context=service_context)
    return query_engine

def process_query(query_engine: RetrieverQueryEngine, query_text: str) -> str:
    response = "{}".format(query_engine.query(query_text))
    return response

def formulate_chat_query(llm: OpenAILLM, context: str, query_text: str) -> str:
    chat_response = llm.chat(messages=[
        ChatMessage(role="system", content=(
            "## YOUR ROLE\n"
            "You are a bioethicist specializing in patient advocacy and human subjects research. Your focus is on interpreting and explaining Informed Consent documents to potential human subjects research participants.\n\n"
            "## RULES\n"
            "- Ensure all responses are directly grounded in the context you are provided.\n"
            "- Responses should be clear and authoritative, delivered in a more formal tone.\n"
            "- Avoid conjunctive adverbs, discourse markers, and both introductory and conclusive statements.\n"
            "- Do not include disclaimers or refer to yourself as an AI.\n"
            "- Provide information in a way that is clear and understandable to potential research participants.\n"
            "- Prioritize accuracy and relevance in your responses. Do not include unnecessary information."
        )),
        ChatMessage(role="user", content=f"RELEVANT STUDY INFORMATION:\n\n{context}"),
        ChatMessage(role="user", content=query_text)
    ])
    return chat_response.message.content

def process_query_concurrently(query_engine: RetrieverQueryEngine, query_text: str) -> str:
    """
    Function to be used with concurrent.futures for processing queries.
    """
    response = "{}".format(query_engine.query(query_text))
    return response

def cleanup_text(text: str) -> str:
    # Remove unwanted characters and sequences
    clean_text = re.sub(r'["`\[\]<>]|---', '', text)
    # Correct extra spaces
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text

def remove_empty_sections(final_responses: Dict[str, str]) -> Dict[str, str]:
    """
    Removes any sections from final_responses that are empty, null, contain only a space,
    or contain only a quoted space.
    """
    filtered_responses = {k: v for k, v in final_responses.items() if v.strip() not in {"", "' '"}}
    return filtered_responses

def process_section(
    section: str,
    queries: List[str],
    query_engine: RetrieverQueryEngine,
    llm: OpenAILLM
) -> Tuple[str, str]:
    context_responses = {}
    final_response = ""

    # If there are multiple queries, process all but the last one concurrently
    if len(queries) > 1:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit concurrent tasks for all but the last query
            futures = [executor.submit(process_query_concurrently, query_engine, query) for query in queries[:-1]]
            for future, query in zip(futures, queries[:-1]):
                try:
                    response = future.result()
                    context_responses[query] = cleanup_text(response)
                except Exception as exc:
                    st.write(f'Query "{query}" generated an exception: {exc}')

    # Combine responses to form the context
    context = "\n\n".join([f"Q: {query}\nA: {response}" for query, response in context_responses.items()])
    cleaned_context = cleanup_text(context)

    # Process the final query using the cleaned context with the chat model
    if queries:
        final_query_text = queries[-1]  # Ensure there's at least one query
        final_response = formulate_chat_query(llm, cleaned_context, final_query_text)
        final_response = cleanup_text(final_response)  # Apply cleanup to the final response

    return section, final_response

def generate_summary(file_path: str) -> str:
    file_path = Path(file_path)
    leaf_nodes = process_documents(file_path)
    service_context, storage_context, vector_store_index = create_service_and_storage_contexts(llm, embed_model, leaf_nodes)
    query_engine = setup_query_engine(service_context, vector_store_index)

    final_responses: Dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(messages_dict)) as executor:
        future_to_section = {executor.submit(process_section, section, queries, query_engine, llm): section for section, queries in messages_dict.items()}
        for future in concurrent.futures.as_completed(future_to_section):
            section = future_to_section[future]
            try:
                section, response = future.result()
                final_responses[section] = response
            except Exception as exc:
                print(f'{section} generated an exception: {exc}')
    
    # Safely insert new predefined entries without overwriting existing ones
    predefined_entries = {
        "section2": "A research study is different from the regular medical care you receive from your doctor. Research studies hope to make discoveries and learn new information about diseases and how to treat them. You should consider the reasons why you might want to join a research study or why it is not the best decision for you at this time.",
        "section3": "Research studies do not always offer the possibility of treating your disease or condition. Research studies also have different kinds of risks and risk levels, depending on the type of the study. You may also need to think about other requirements for being in the study. For example, some studies require you to travel to scheduled visits at the study site in Ann Arbor or elsewhere. This may require you to arrange travel, change work schedules, find child care, or make other plans. In your decision to participate in this study, consider all of these matters carefully."
    }
    for key, value in predefined_entries.items():
        if key not in final_responses:  # Ensure not to overwrite processed responses
            final_responses[key] = value

    # Remove sections that are empty, null, only a space, or only a quoted space
    final_responses = remove_empty_sections(final_responses)

    # Concatenate final responses into a single string separated by two newlines
    combined_responses = "\n\n".join(final_responses[section] for section in sorted(final_responses))
    return combined_responses

if __name__ == "__main__":
    if len(argv) > 1:
        print(generate_summary(argv[1]))
