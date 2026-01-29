import logging
import os
import faiss
import shutil
import sys
import json
import time
from tqdm import tqdm
import copy
import argparse
from typing import Any, Dict, List, Optional
from pymongo import MongoClient
from llama_index.core import StorageContext, Settings, Document, VectorStoreIndex, load_index_from_storage
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.llms.openai import OpenAI
import sys

sys.path.append('.')
from tasks.common import trace_langfuse
from tasks.kilt_utils import normalize_answer

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# 从段落列表提取摘要（标题+段首）
def get_abstract(paragraphs: List[str]) -> str:
    res = []
    for p in paragraphs:  # paragraphs[0] is the title
        if p.startswith('Section::::') or p.startswith('BULLET::::'):
            break
        res.append(p)
    return '\n'.join(res).strip()


def get_nodes(max_documents):
    mongo_client = MongoClient("mongodb://localhost:27017/")
    db = mongo_client["nba-datalake"]["wiki-documents"]
    num_docs = db.count_documents({})
    nodes = []
    for page in tqdm(db.find(), total=num_docs):
        if max_documents and len(nodes) > max_documents:
            break
        # 作为元数据
        wikipedia_id = page["wikipedia_id"]
        wikipedia_title = page["wikipedia_title"]
        page_category = page["categories"]

        abstract = get_abstract(page["text"])

        if not abstract:
            print('Warning: empty abstract for', wikipedia_title)

        node = TextNode(
            text=abstract,
            metadata={
                "wikipedia_id": wikipedia_id,
                "wikipedia_title": wikipedia_title,
                "categories": page_category,
            },
            excluded_llm_metadata_keys=["wikipedia_id", "categories"],
            excluded_embed_metadata_keys=["wikipedia_id", "categories"],
            metadata_seperator="::",
            metadata_template="{key}=>{value}",
            text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
        )
        nodes.append(node)
    return nodes

# 索引构建
def get_doc_index(emb_model, index_type="default", max_documents=None):
    assert index_type == "default"
    # 嵌入
    if emb_model.startswith('text-embedding'):
        Settings.embed_model = OpenAIEmbedding(model=emb_model, embed_batch_size=1000)
    else:
        Settings.embed_model = HuggingFaceEmbedding(emb_model)
    persist_dir = os.path.join("indices", 'doc_' + index_type + '_' + emb_model.replace('/', '--'))

    if not (os.path.exists(persist_dir) and os.listdir(persist_dir)):
        t0 = time.time()
        nodes = get_nodes(max_documents=max_documents)
        created_index = VectorStoreIndex(nodes, show_progress=True)
        created_index.storage_context.persist(persist_dir=persist_dir)
        print(f"Index created in {time.time() - t0:.2f} seconds")

    t0 = time.time()
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
    print(f"Index loaded in {time.time() - t0:.2f} seconds")
    return index


def get_doc_query_engine(
        llm, retriever, doc_top_k, index_type="default", max_documents=None,
):
    openai_base_url = os.environ.get("OPENAI_BASE_URL")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    print(f"[debug] OPENAI_BASE_URL={openai_base_url}")
    print(f"[debug] OPENAI_API_KEY={'set' if openai_api_key else 'missing'}")

    Settings.llm = OpenAI(
        temperature=0,
        model=llm,
        api_base=openai_base_url,   # 关键：显式指定
        api_key=openai_api_key,     # 关键：显式指定
    )
    
    
    # Settings.llm = OpenAI(temperature=0, model=llm)
    if retriever.lower() == "bm25":   # 关键词检索
        nodes = get_nodes(max_documents)
        retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=doc_top_k)
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
        )
    else:   # 向量检索
        index = get_doc_index(retriever, index_type, max_documents)

        query_engine = index.as_query_engine(
            similarity_top_k=doc_top_k,
            verbose=True,
        )
    return query_engine


@trace_langfuse(name="doc_discovery")
def get_responses(engine, dataset) -> List[dict]:
    all_response = []
    for d in dataset:
        response = engine.query(d["question"])
        d = copy.deepcopy(d)
        d['model_response'] = str(response)
        d['model_provenance'] = {
            'docs': [{
                'wikipedia_title': d1['wikipedia_title'],
            } for d1 in response.metadata.values()]
        }
        all_response.append(d)
    return all_response


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    return len(set(retrieved[:k]) & set(relevant)) / k


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    return len(set(retrieved[:k]) & set(relevant)) / len(relevant)


def r_precision(retrieved: list[str], relevant: list[str]) -> float:
    return precision_at_k(retrieved, relevant, len(relevant)) if relevant else 0.0


def evaluate(all_response: List[dict]) -> dict:
    res = {
        'metrics': {},
        'responses': []
    }
    ks = [1, 2, 3, 5, 10, 20]
    # max_k = max(ks)
    # assert max_k <= all(max_k <= len(d['model_provenance']['spans']) for d in all_response)

    for d in all_response:
        d = copy.deepcopy(d)

        # Compute accuracy
        d['metric_accuracy'] = float(normalize_answer(d["answer"]) in normalize_answer(d["model_response"]))

        # Compute retrieval metrics
        retrieved = [doc['wikipedia_title'] for doc in d['model_provenance']['docs']]
        relevant = list(set(span['wikipedia_title'] for span in d['provenance_doc']['paragraphs']))
        d['metric_r_precision'] = r_precision(retrieved, relevant)
        for k in ks:
            d[f'metric_precision@{k}'] = precision_at_k(retrieved, relevant, k)
        for k in ks:
            d[f'metric_recall@{k}'] = recall_at_k(retrieved, relevant, k)

        res['responses'].append(d)

    metrics = ['accuracy', 'r_precision'] + [f'precision@{k}' for k in ks] + [f'recall@{k}' for k in ks]
    for metric in metrics:
        res['metrics'][metric] = sum(d[f'metric_{metric}'] for d in res['responses']) / len(res['responses'])

    r_precision_by_category = {
        'single': [],
        'multi': [],
    }
    for d in res['responses']:
        relevant = list(set(span['wikipedia_title'] for span in d['provenance_doc']['paragraphs']))
        if len(relevant) == 1:
            r_precision_by_category['single'].append(d['metric_r_precision'])
        else:
            r_precision_by_category['multi'].append(d['metric_r_precision'])
    res['metrics']['r_precision_single'] = sum(r_precision_by_category['single']) / len(r_precision_by_category['single'])
    res['metrics']['r_precision_multi'] = sum(r_precision_by_category['multi']) / len(r_precision_by_category['multi'])
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="graph", choices=["graph", "doc", "table", "all"])
    parser.add_argument('--inputs', default=["benchmark/q_doc_single1.json"], nargs="+")
    parser.add_argument('--output_dir', default='outputs/test_doc_discovery/')
    parser.add_argument('--overwrite', action="store_true")

    # parameters for mode=doc
    parser.add_argument('--llm', default="gpt-3.5-turbo")
    parser.add_argument('--doc_top_k', default=20, type=int)
    # parser.add_argument('--chunk_size', default=512, type=int)
    parser.add_argument('--index_type', default="default", choices=["default", "faiss", "duckdb"])
    parser.add_argument('--retriever', default="BAAI/bge-base-en-v1.5")

    # 新增
    parser.add_argument('--openai_api_key', default=None, help='覆盖 OPENAI_API_KEY')
    parser.add_argument('--openai_base_url', default=None, help='覆盖 OPENAI_BASE_URL，例如 https://api.chatanywhere.tech/v1')

    args = parser.parse_args()
    print(args)
    print()

    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
    if args.openai_base_url:
        os.environ["OPENAI_BASE_URL"] = args.openai_base_url


    if args.overwrite and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    response_output_path = os.path.join(args.output_dir, "responses.json")
    if not os.path.exists(response_output_path):
        # Get query engine
        engine = get_doc_query_engine(
            llm=args.llm,
            retriever=args.retriever,
            doc_top_k=args.doc_top_k,
            index_type=args.index_type,
            max_documents=None
        )
        # Load dataset
        dataset = []
        for path in args.inputs:
            with open(path) as f:
                dataset += json.load(f)
        # Run queries
        all_response = get_responses(engine, dataset)
        with open(response_output_path, "w") as f:
            json.dump(all_response, f, indent=2)
        print(f'Responses saved to {response_output_path}')

    with open(response_output_path) as f:
        all_response = json.load(f)
    print(f'Loaded {len(all_response)} responses from {response_output_path}')

    print("\n" + "="*50)
    print("生成的答案：")
    print("="*50)
    for i, resp in enumerate(all_response, 1):
        print(f"\n问题 {i}: {resp['question']}")
        print(f"答案: {resp.get('model_response', 'N/A')}")
        print(f"检索到的文档: {[doc.get('wikipedia_title', 'N/A') for doc in resp.get('model_provenance', {}).get('docs', [])]}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
