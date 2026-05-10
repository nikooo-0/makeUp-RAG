# -*- coding: utf-8 -*-
"""
化妆品知识库问答系统 - 主程序
"""
import os
import sys
import io

# 设置标准输出编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from embeddings import get_embedder
from vectorstore import CosmeticVectorStore


# 是否使用模拟向量（本地测试用）
# 设置为False需要配置DashScope API Key
USE_MOCK_EMBEDDING = False


class CosmeticQASystem:
    """化妆品问答系统"""

    def __init__(self):
        self.vector_store = None
        self.embedder = None

    def initialize(self, data_path: str = None, use_mock: bool = False):
        """初始化系统

        Args:
            data_path: 数据文件路径
            use_mock: False使用DashScope, True使用模拟向量模型
        """
        print("=" * 50)
        print("初始化化妆品知识库问答系统...")
        print("=" * 50)

        # 数据文件路径
        if data_path is None:
            data_path = config.DATA_FILE

        # 初始化向量化模型
        print("\n[1/3] 初始化向量化模型...")
        self.embedder = get_embedder(use_mock=use_mock)
        print("向量化模型初始化完成")

        # 初始化向量存储
        print("\n[2/3] 初始化向量存储...")
        self.vector_store = CosmeticVectorStore(embedding_dim=config.EMBEDDING_DIM)

        # 加载数据
        print(f"\n[3/3] 加载数据从: {data_path}")
        data = self.vector_store.load_csv_data(data_path)
        print(f"成功加载 {len(data)} 条成分数据")

        # 准备文档
        documents = self.vector_store.prepare_documents(data)
        print(f"成功处理 {len(documents)} 条文档")

        # 创建向量存储
        self.vector_store.create_vector_store(documents, self.embedder)

        # 保存向量库
        save_path = "C:/Users/niko/Desktop/cosmetic_qa_data/vectorstore"
        self.vector_store.save(save_path)

        print("\n" + "=" * 50)
        print("系统初始化完成!")
        print("=" * 50)

    def query(self, question: str, top_k: int = 5) -> str:
        """问答"""
        if self.vector_store is None:
            return "请先初始化系统"

        print(f"\n提问: {question}")

        # 搜索相似内容
        results = self.vector_store.similarity_search(question, top_k=top_k)

        if not results:
            return "未找到相关内容"

        # 构建回答
        answer = self._build_answer(question, results)

        return answer

    def _build_answer(self, question: str, results: list) -> str:
        """构建回答"""
        # 判断问题类型
        question_lower = question.lower()

        # 根据问题类型，构建不同格式的回答
        if any(keyword in question_lower for keyword in ["推荐", "适合", "用什么", "哪种"]):
            # 推荐类问题
            answer = "根据您的需求，为您推荐以下成分:\n\n"
            for i, r in enumerate(results, 1):
                answer += f"{i}. {r['ingredient_name']}\n"
                answer += f"   功效: {r['efficacy_category']}\n"
                answer += f"   适用: {r['skin_type']}\n"
                answer += f"   说明: {r['description']}\n"
                if r['precautions']:
                    answer += f"   注意: {r['precautions']}\n"
                answer += "\n"

        elif any(keyword in question_lower for keyword in ["合规", "安全", "有没有", "是否"]):
            # 合规类问题
            answer = "关于成分安全性:\n\n"
            for r in results:
                answer += f"• {r['ingredient_name']}: {r['description']}\n"
                if r['precautions']:
                    answer += f"  注意事项: {r['precautions']}\n"

        elif any(keyword in question_lower for keyword in ["功效", "作用", "有什么用"]):
            # 功效类问题
            answer = "相关成分的功效说明:\n\n"
            for r in results:
                answer += f"• {r['ingredient_name']}: {r['efficacy_category']}\n"
                answer += f"  说明: {r['description']}\n"

        else:
            # 通用问答
            answer = "为您找到以下相关信息:\n\n"
            for i, r in enumerate(results, 1):
                answer += f"{i}. {r['ingredient_name']}\n"
                answer += f"   功效: {r['efficacy_category']}\n"
                answer += f"   适用肤质: {r['skin_type']}\n"
                answer += f"   说明: {r['description']}\n"
                answer += "\n"

        return answer


def main():
    """主函数"""
    # 创建问答系统实例
    qa_system = CosmeticQASystem()

    # 初始化系统
    qa_system.initialize()

    # 示例问答
    print("\n" + "=" * 50)
    print("开始问答测试...")
    print("=" * 50)

    # 测试问题
    test_questions = [
        "我想要美白，推荐什么成分？",
        "敏感肌可以用什么成分？",
        "烟酰胺安全吗？",
        "抗衰老有哪些成分？",
    ]

    for question in test_questions:
        answer = qa_system.query(question)
        print(answer)
        print("-" * 30)


if __name__ == "__main__":
    main()