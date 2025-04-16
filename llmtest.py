import ollama

response = ollama.chat(
    model='llama3.2-vision',
    messages=[{
        'role': 'user',
        'content': 'What is in this image?',
        'images': ['/media/li-ruiqin/5D4A4FF551440BFB/workplace/notes/image/image-15.png']
    }]
)

print(response)