# -*- coding: utf-8 -*-
"""
向量存储模块 - 同时支持FAISS(本地)和Milvus(云端)
"""
import os
import csv
import numpy as np
from typing import List, Dict, Any

# 优先使用FAISS本地存储
USE_FAISS = True  # 设置为False可切换到Milvus

if USE_FAISS:
    import faiss
    from langchain_community.vectorstores import FAISS
else:
    from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
    import config as sys_config


class Document:
    """简单文档类"""
    def __init__(self, page_content: str, metadata: Dict[str, Any]):
        self.page_content = page_content
        self.metadata = metadata


class CosmeticVectorStore:
    """化妆品知识库向量存储"""

    def __init__(self, embedding_dim: int = 1536):
        self.embedding_dim = embedding_dim
        self.vector_store = None
        self.documents = []
        self.ids = []

    def load_csv_data(self, csv_path: str) -> List[Dict]:
        """加载CSV数据"""
        data = []
        # 尝试不同编码
        encodings = ['utf-8-sig', 'utf-8', 'gbk']

        for encoding in encodings:
            try:
                with open(csv_path, 'r', encoding=encoding) as f:
                    reader = csv.DictReader(f)
                    data = list(reader)
                print(f"使用{encoding}编码成功加载{len(data)}条数据")
                break
            except UnicodeDecodeError:
                continue

        if not data:
            raise Exception("CSV文件读取失败，请检查文件编码")

        return data

    def prepare_documents(self, data: List[Dict]) -> List[Document]:
        """准备文档"""
        documents = []
        for i, item in enumerate(data):
            # 构建文档内容
            content = f"""成分名称: {item.get('成分名称', '')}
功效分类: {item.get('功效分类', '')}
适用肤质: {item.get('适用肤质', '')}
注意事项: {item.get('注意事项', '')}
简要说明: {item.get('简要说明', '')}"""

            doc = Document(
                page_content=content,
                metadata={
                    "ingredient_name": item.get('成分名称', ''),
                    "efficacy_category": item.get('功效分类', ''),
                    "skin_type": item.get('适用肤质', ''),
                    "precautions": item.get('注意事项', ''),
                    "description": item.get('简要说明', '')
                }
            )
            documents.append(doc)

        return documents

    def create_vector_store(self, documents: List[Document], embedding_model):
        """创建向量存储"""
        if USE_FAISS:
            # 使用FAISS
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]

            self.vector_store = FAISS.from_texts(
                texts=texts,
                embedding=embedding_model,
                metadatas=metadatas
            )
            print(f"FAISS向量存储创建成功，共{len(documents)}条文档")
        else:
            # 使用Milvus - 暂不支持
            pass

    def similarity_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """相似度搜索"""
        if self.vector_store is None:
            raise Exception("向量存储未初始化")

        docs = self.vector_store.similarity_search(query, k=top_k)

        results = []
        for doc in docs:
            results.append({
                "ingredient_name": doc.metadata.get("ingredient_name", ""),
                "efficacy_category": doc.metadata.get("efficacy_category", ""),
                "skin_type": doc.metadata.get("skin_type", ""),
                "precautions": doc.metadata.get("precautions", ""),
                "description": doc.metadata.get("description", "")
            })

        return results

    def save(self, path: str):
        """保存向量存储"""
        if self.vector_store is None:
            raise Exception("向量存储未初始化")

        if USE_FAISS:
            self.vector_store.save_local(path)
            print(f"向量存储已保存到: {path}")

    def load(self, path: str, embedding_model):
        """加载向量存储"""
        if USE_FAISS:
            self.vector_store = FAISS.load_local(
                path,
                embedding_model,
                allow_dangerous_deserialization=True
            )
            print(f"向量存储已从: {path} 加载")


# ===== 以下是Milvus相关函数 (生产环境使用) =====

def init_milvus_connection(host: str = "localhost", port: int = 19530):
    """初始化Milvus连接"""
    connections.connect(host=host, port=port)
    print("Milvus连接成功")


def create_milvus_collection(name: str, embedding_dim: int = 1536):
    """创建Milvus Collection"""
    if utility.has_collection(name):
        utility.drop_collection(name)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="ingredient_name", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="efficacy_category", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="skin_type", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="precautions", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim)
    ]

    schema = CollectionSchema(fields=fields, description="化妆品成分知识库")
    collection = Collection(name=name, schema=schema)

    # 创建索引
    index_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
    collection.create_index(field_name="vector", index_params=index_params)

    print(f"Milvus Collection创建成功: {name}")
    return collection


def milvus_search(query_vector: List[float], collection_name: str, top_k: int = 5):
    """Milvus相似度搜索"""
    collection = Collection(collection_name)
    collection.load()

    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

    results = collection.search(
        data=[query_vector],
        anns_field="vector",
        param=search_params,
        limit=top_k,
        output_fields=["ingredient_name", "efficacy_category", "skin_type", "precautions", "description"]
    )

    return results