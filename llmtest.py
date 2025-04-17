import ollama

response = ollama.chat(
    model='llama3.2-vision:11b',
    messages=[{
        'role': 'user',
        'content': '这张图片描述了什么',
        'images': ['/media/li-ruiqin/5D4A4FF551440BFB/workplace/notes/image/image-15.png']
    }]
)

print(response)
