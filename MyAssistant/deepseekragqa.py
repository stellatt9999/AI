# 导入库模块
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings  # 若DeepSeek提供Embedding API，可替换为对应类
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 导入自定义模块
from deepseekqa import deepseek_qa

# 1. 加载并拆分本地文档
loader = TextLoader("my个人信息.txt", encoding="utf-8")  # 替换为你的文件路径
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(documents)

# 2. 初始化Embedding模型（这里用开源模型示例，若DeepSeek有Embedding API可替换）
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")  # 中文Embedding模型

# 3. 创建向量库
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# 4. 定义RAG链：用DeepSeek模型作为LLM，结合检索结果
def deepseek_rag_qa(question, vectorstore, api_key):
    # 检索相关文档（前3条最相关）
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    context_docs = retriever.get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in context_docs])
    
    # 构造带上下文的提示词
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="基于以下上下文回答问题：\n{context}\n问题：{question}\n回答："
    )
    full_prompt = prompt.format(context=context, question=question)
    print(f"full_prompt:{full_prompt}" )

    # 调用DeepSeek API生成回答
    return deepseek_qa(full_prompt, api_key)
