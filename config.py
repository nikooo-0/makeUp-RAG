# -*- coding: utf-8 -*-
"""
配置文件
"""
import os

# DashScope配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-8a4d28a1c1584406bdbab6f6de8c0533")

# Milvus配置
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
COLLECTION_NAME = "cosmetic_knowledge"

# 向量模型配置
EMBEDDING_MODEL = "text-embedding-v1"
EMBEDDING_DIM = 1536

# 数据文件路径
DATA_FILE = "C:/Users/niko/Desktop/makeUp-RAG/成分功效数据.csv"

# 日志配置
LOG_LEVEL = "INFO"