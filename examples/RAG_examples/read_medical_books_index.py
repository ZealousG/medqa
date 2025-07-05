import os
import json
import pickle

INDEX_DIR = "data/indices/medical_books"


def main():
    # 读取config.json
    config_path = os.path.join(INDEX_DIR, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    print("config.json 内容:")
    print(json.dumps(config, ensure_ascii=False, indent=2))
    print()

    # 读取document_ids.json
    doc_ids_path = os.path.join(INDEX_DIR, "document_ids.json")
    with open(doc_ids_path, "r", encoding="utf-8") as f:
        doc_ids = json.load(f)
    print(f"document_ids.json 共 {len(doc_ids)} 个ID，前5个:")
    print(doc_ids[:5])
    print()

    # 读取documents.pkl
    docs_path = os.path.join(INDEX_DIR, "documents.pkl")
    with open(docs_path, "rb") as f:
        docs = pickle.load(f)
    print(f"documents.pkl 共 {len(docs)} 个文档块，前2个类型和内容:")
    for i, doc in enumerate(docs[:2]):
        print(f"第{i+1}个类型: {type(doc)}")
        if hasattr(doc, 'page_content'):
            print(f"内容片段: {doc.page_content[:100]}...")
        else:
            print(f"内容: {str(doc)[:100]}...")
        if hasattr(doc, 'metadata'):
            print(f"元数据: {doc.metadata}")
        print()

if __name__ == "__main__":
    main() 