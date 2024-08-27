import os
import re
from typing import List, Union

import tiktoken
from langchain_community.document_loaders import UnstructuredFileLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter


def num_tokens(text: str, model: str = 'gpt-3.5-turbo-0613') -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


SENTENCE_SIZE = 100
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n", ".", "。", "!", "！", "?", "？", "；", ";", "……", "…", "、", "，", ",", " "],
    chunk_size=400,
    length_function=num_tokens,
)


# async def insert_files_to_milvus(self, user_id, kb_id, local_files: List[LocalFile]):
#     debug_logger.info(f'insert_files_to_milvus: {kb_id}')
#     milvus_kv = self.match_milvus_kb(user_id, [kb_id])
#     assert milvus_kv is not None
#     success_list = []
#     failed_list = []
#
#     for local_file in local_files:
#         start = time.time()
#         try:
#             local_file.split_file_to_docs(self.get_ocr_result)
#             content_length = sum([len(doc.page_content) for doc in local_file.docs])
#         except Exception as e:
#             error_info = f'split error: {traceback.format_exc()}'
#             debug_logger.error(error_info)
#             self.milvus_summary.update_file_status(local_file.file_id, status='red')
#             failed_list.append(local_file)
#             continue
#         end = time.time()
#         self.milvus_summary.update_content_length(local_file.file_id, content_length)
#         debug_logger.info(f'split time: {end - start} {len(local_file.docs)}')
#         start = time.time()
#         try:
#             local_file.create_embedding()
#         except Exception as e:
#             error_info = f'embedding error: {traceback.format_exc()}'
#             debug_logger.error(error_info)
#             self.milvus_summary.update_file_status(local_file.file_id, status='red')
#             failed_list.append(local_file)
#             continue
#         end = time.time()
#         debug_logger.info(f'embedding time: {end - start} {len(local_file.embs)}')
#
#         self.milvus_summary.update_chunk_size(local_file.file_id, len(local_file.docs))
#         ret = await milvus_kv.insert_files(local_file.file_id, local_file.file_name, local_file.file_path,
#                                            local_file.docs, local_file.embs)
#         insert_time = time.time()
#         debug_logger.info(f'insert time: {insert_time - end}')
#         if ret:
#             self.milvus_summary.update_file_status(local_file.file_id, status='green')
#             success_list.append(local_file)
#         else:
#             self.milvus_summary.update_file_status(local_file.file_id, status='yellow')
#             failed_list.append(local_file)
#     debug_logger.info(
#         f"insert_to_milvus: success num: {len(success_list)}, failed num: {len(failed_list)}")


class LocalFile:
    def __init__(self, file: str, file_id, file_name):
        self.file_id = file_id
        self.docs: List[Document] = []
        self.embs = []
        self.url = None
        self.file_name = file_name
        if isinstance(file, str):
            self.file_path = file
            with open(file, 'rb') as f:
                self.file_content = f.read()

    def split_file_to_docs(self, sentence_size=SENTENCE_SIZE):
        if self.file_path.lower().endswith(".md"):
            loader = UnstructuredFileLoader(self.file_path, mode="elements")
            docs = loader.load()
        elif self.file_path.lower().endswith(".txt"):
            loader = TextLoader(self.file_path, autodetect_encoding=True)
            texts_splitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
            docs = loader.load_and_split(texts_splitter)
        else:
            raise TypeError("文件类型不支持，目前仅支持：[md,txt,pdf,jpg,png,jpeg,docx,xlsx,pptx,eml,csv]")

        # 重构docs，如果doc的文本长度大于800tokens，则利用text_splitter将其拆分成多个doc
        # text_splitter: RecursiveCharacterTextSplitter
        docs = text_splitter.split_documents(docs)

        # 这里给每个docs片段的metadata里注入file_id
        for doc in docs:
            doc.metadata["file_id"] = self.file_id
            doc.metadata["file_name"] = self.url if self.url else os.path.split(self.file_path)[-1]
        write_check_file(self.file_path, docs)
        self.docs = docs

    def create_embedding(self):
        self.embs = self.emb_infer._get_len_safe_embeddings([doc.page_content for doc in self.docs])


class ChineseTextSplitter(CharacterTextSplitter):
    def __init__(self, pdf: bool = False, sentence_size: int = SENTENCE_SIZE, **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf
        self.sentence_size = sentence_size

    def split_text(self, text: str) -> List[str]:
        if self.pdf:
            text = re.sub(r"\n{3,}", r"\n", text)
            text = re.sub('\s', " ", text)
            text = re.sub("\n\n", "", text)

        text = re.sub(r'([;；.!?。！？\?])([^”’])', r"\1\n\2", text)  # 单字符断句符
        text = re.sub(r'(\.{6})([^"’”」』])', r"\1\n\2", text)  # 英文省略号
        text = re.sub(r'(\…{2})([^"’”」』])', r"\1\n\2", text)  # 中文省略号
        text = re.sub(r'([;；!?。！？\?]["’”」』]{0,2})([^;；!?，。！？\?])', r'\1\n\2', text)
        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        text = text.rstrip()  # 段尾如果有多余的\n就去掉它
        # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
        ls = [i for i in text.split("\n") if i]
        for ele in ls:
            if len(ele) > self.sentence_size:
                ele1 = re.sub(r'([,，.]["’”」』]{0,2})([^,，.])', r'\1\n\2', ele)
                ele1_ls = ele1.split("\n")
                for ele_ele1 in ele1_ls:
                    if len(ele_ele1) > self.sentence_size:
                        ele_ele2 = re.sub(r'([\n]{1,}| {2,}["’”」』]{0,2})([^\s])', r'\1\n\2', ele_ele1)
                        ele2_ls = ele_ele2.split("\n")
                        for ele_ele2 in ele2_ls:
                            if len(ele_ele2) > self.sentence_size:
                                ele_ele3 = re.sub('( ["’”」』]{0,2})([^ ])', r'\1\n\2', ele_ele2)
                                ele2_id = ele2_ls.index(ele_ele2)
                                ele2_ls = ele2_ls[:ele2_id] + [i for i in ele_ele3.split("\n") if i] + ele2_ls[
                                                                                                       ele2_id + 1:]
                        ele_id = ele1_ls.index(ele_ele1)
                        ele1_ls = ele1_ls[:ele_id] + [i for i in ele2_ls if i] + ele1_ls[ele_id + 1:]

                id = ls.index(ele)
                ls = ls[:id] + [i for i in ele1_ls if i] + ls[id + 1:]
        return ls


def write_check_file(filepath, docs):
    folder_path = os.path.join(os.path.dirname(filepath), "tmp_files")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    fp = os.path.join(folder_path, 'load_file.txt')
    with open(fp, 'a+', encoding='utf-8') as fout:
        fout.write("filepath=%s,len=%s" % (filepath, len(docs)))
        fout.write('\n')
        for i in docs:
            fout.write(str(i))
            fout.write('\n')
        fout.close()


if __name__ == '__main__':
    local_file = LocalFile("../dataset/company_low.txt", "file_1", "low")
    local_file.split_file_to_docs()