from flask import Flask, render_template_string, request

app = Flask(__name__)

# ------------------------------
# 导入业务逻辑（基于文档生成回答）
# ------------------------------
from deepseekragqa import vectorstore, deepseek_rag_qa

# ------------------------------
# 导入API_KEY
# ------------------------------
from common.conf import API_KEY


# ------------------------------
# Flask 路由与页面
# ------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    # 定义网页模板（HTML）
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>我的 Flask 网页</title>
        <style>
            body { max-width: 800px; margin: 0 auto; padding: 20px; font-family: Arial; }
            .container { text-align: center; margin-top: 50px; }
            input { width: 60%; padding: 8px; margin: 10px 0; }
            button { padding: 8px 20px; background: #4CAF50; color: white; border: none; cursor: pointer; }
            .result { margin-top: 20px; padding: 10px; border: 1px solid #ddd; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>个人信息查询助手</h1>
            <form method="POST">
                <input type="text" name="user_input" placeholder="请输入内容..." 
                       value="{{ user_input if user_input else '' }}">
                <button type="submit">提交处理</button>
            </form>
            {% if result %}
                <div class="result">{{ result }}</div>
            {% endif %}
        </div>
    </body>
    </html>
    """
    
    # 处理POST请求（用户提交数据时）
    if request.method == "POST":
        user_input = request.form.get("user_input", "").strip()  # 获取用户输入
        result = deepseek_rag_qa(user_input, vectorstore, API_KEY)  # 调用Python逻辑处理
        # 渲染页面并返回结果
        return render_template_string(html_template, user_input=user_input, result=result)
    
    # 处理GET请求（首次访问页面时）
    return render_template_string(html_template)


if __name__ == "__main__":
    app.run(debug=True)  # 启动服务器，debug=True 表示开发模式（自动刷新）



# 附：查看端口占用（调试用）
# netstat -ano | findstr :5000
