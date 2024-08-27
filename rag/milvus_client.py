from typing import List

from pymilvus import utility, connections, FieldSchema, CollectionSchema, DataType, Collection, Partition
import random

connections.connect("default", host="localhost", port="19530")


class MilvusClient:

    def __init__(self):
        self.search_params = {"metric_type": "L2", "params": {"nprobe": 256}}
        self.collection = None
        self.partitions: List[Partition] = []
        self.collection_name = "my_collection"
        self.init()

    def init(self):
        try:
            collection_name = self.collection_name
            connections.connect(host="localhost", port="19530", user="root", password="123456", db_name="default")
            if utility.has_collection(collection_name):
                self.collection = Collection(collection_name)

            else:
                fields = [
                    FieldSchema(name='chunk_id', dtype=DataType.VARCHAR, max_length=64, is_primary=True),
                    FieldSchema(name='file_id', dtype=DataType.VARCHAR, max_length=64),
                    FieldSchema(name='file_name', dtype=DataType.VARCHAR, max_length=640),
                    FieldSchema(name='file_path', dtype=DataType.VARCHAR, max_length=640),
                    FieldSchema(name='timestamp', dtype=DataType.VARCHAR, max_length=64),
                    FieldSchema(name='content', dtype=DataType.VARCHAR, max_length=4000),
                    FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=768)
                ]
                self.collection = Collection(name=collection_name, schema=CollectionSchema(fields))

                self.collection.create_index(field_name="embedding", index_params={"metric_type": "L2", "index_type": "GPU_IVF_FLAT", "params": {"nlist": 2048}})
            for kb_id in self.kb_ids:
                if not self.collection.has_partition(kb_id):
                    self.collection.create_partition(kb_id)
            self.partitions = [Partition(self.collection, kb_id) for kb_id in self.kb_ids]

            self.collection.load()
        except Exception as e:
            raise e


if __name__ == '__main__':
    connections.connect(host="localhost", port="19530", user="root", password="123456", db_name="default")

    # 定义字段
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128)
    ]
    # 定义集合模式
    schema = CollectionSchema(fields, description="测试集合")

    # 创建集合
    collection = Collection(name="collection_1", schema=schema)

    # 创建索引
    index_params = {
        "index_type": "IVF_FLAT",
        "params": {"nlist": 100},
        "metric_type": "L2"
    }
    collection.create_index(field_name="embedding", index_params=index_params)

    # 生成随机向量数据
    vectors = [[random.random() for _ in range(128)] for _ in range(10)]
    # 插入数据
    collection.insert([vectors])
