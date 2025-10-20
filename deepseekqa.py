import requests
import json

def deepseek_qa(question, api_key):
    # DeepSeek API端点（具体以官方文档为准）
    url = "https://api.deepseek.com/v1/chat/completions"
    
    # 请求头：包含API密钥认证
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # 请求体：定义模型、对话历史（此处仅单轮问答）
    data = {
        "model": "deepseek-chat",  # 模型名称
        "messages": [
            {"role": "user", "content": question}  # 用户问题
        ],
        "temperature": 0.7,  # 生成多样性（0-1，值越高越随机）
        "max_tokens": 512  # 最大回答长度
    }
    
    try:
        # 发送POST请求
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # 检查请求是否成功
        # 解析返回结果（提取回答内容）
        result = response.json()
        answer = result["choices"][0]["message"]["content"]
        return answer
    except Exception as e:
        return f"调用失败：{str(e)}"
