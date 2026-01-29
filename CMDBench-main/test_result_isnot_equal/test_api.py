from openai import OpenAI

# ✅ 直接写死测试
client = OpenAI(
    api_key="sk-FGHIXlyPYpUGzovjKzG7UYv7J7vfJYevqKsEf8o3EryiuiCA",  # 替换为你的真实密钥
    base_url="https://api.chatanywhere.tech/v1"
)

print("=== 测试连接 ===")
try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=10
    )
    print("✅ 连接成功！")
    print(f"回复: {response.choices[0].message.content}")
except Exception as e:
    print(f"❌ 连接失败: {e}")
