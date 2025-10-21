# ------------------------------
# 导入业务逻辑（基于文档生成回答）
# ------------------------------
from deepseekragqa import vectorstore, deepseek_rag_qa

# ------------------------------
# 导入API_KEY
# ------------------------------
from common.conf import API_KEY


# 测试：基于本地文档回答
#print(deepseek_rag_qa("兴趣爱好是什么？", vectorstore, API_KEY))


# 交互测试
while True:
    user_input = input("")
    if user_input in ["退出", "再见"]:
        print("助手：再见！")
        break
    answer = deepseek_rag_qa(user_input, vectorstore, API_KEY)
    print(f"助手：{answer}")
