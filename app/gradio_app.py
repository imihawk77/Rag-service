import gradio as gr
import requests
import json

API_URL = "http://localhost:8000"

def upload_file(file):
    """
    Загрузка PDF файла через Gradio.
    """
    if file is None:
        return "Пожалуйста, выберите PDF файл"
    
    try:
        files = {'file': open(file.name, 'rb')}
        response = requests.post(f"{API_URL}/upload", files=files)
        
        if response.status_code == 200:
            data = response.json()
            return (
                f" Успешно загружено!\n"
                f" Создано чанков: {data['chunks_count']}\n"
                f" Документы: {', '.join(data['documents'])}"
            )
        else:
            return f" Ошибка: {response.text}"
    except Exception as e:
        return f" Ошибка подключения к серверу: {str(e)}"

def upload_file_with_progress(file, progress=gr.Progress()):
    """
    Загрузка PDF файла с индикатором прогресса.
    """
    if file is None:
        return " Пожалуйста, выберите PDF файл"
    
    try:
        progress(0, desc=" Начинаем загрузку...")
        
        files = {'file': open(file.name, 'rb')}
        progress(0.3, desc=" Отправка файла на сервер...")
        
        response = requests.post(f"{API_URL}/upload", files=files)
        
        progress(0.7, desc=" Обработка и индексация...")
        
        if response.status_code == 200:
            data = response.json()
            progress(1.0, desc=" Готово!")
            return (
                f" Успешно загружено!\n"
                f" Создано чанков: {data['chunks_count']}\n"
                f" Документы: {', '.join(data['documents'])}"
            )
        else:
            progress(1.0, desc=" Ошибка!")
            return f" Ошибка: {response.text}"
    except Exception as e:
        progress(1.0, desc=" Ошибка!")
        return f" Ошибка подключения к серверу: {str(e)}"

def ask_question(message, history):
    """
    Задать вопрос по документам (для Chatbot).
    """
    if not message:
        return history
    
    try:
        payload = {"question": message, "top_k": 5}
        response = requests.post(f"{API_URL}/query", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            answer = data['answer']
            
            # Добавляем источники
            sources_text = "\n\n **Источники:**\n"
            for i, source in enumerate(data['sources'], 1):
                sources_text += f"\n{i}. {source['source']} (релевантность: {source['score']:.3f})"
                sources_text += f"\n   ...{source['text'][:150]}...\n"
            
            full_response = f"{answer}\n\n{sources_text}"
        else:
            full_response = f" Ошибка: {response.text}"
    except Exception as e:
        full_response = f" Ошибка подключения к серверу: {str(e)}"
    
    # Добавляем сообщения в историю в правильном формате
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": full_response})
    
    return history

def check_status():
    """
    Проверка статуса сервера и загруженных документов.
    """
    try:
        response = requests.get(f"{API_URL}/status")
        if response.status_code == 200:
            data = response.json()
            if data['documents_loaded']:
                return (
                    f" Сервер работает\n"
                    f" Загружено чанков: {data['chunks_count']}\n"
                    f" Готов к вопросам: {data['is_ready']}"
                )
            else:
                return " Сервер работает, но документы не загружены. Загрузите PDF!"
        else:
            return f" Ошибка: {response.text}"
    except Exception as e:
        return f" Сервер не доступен. Запустите API командой:\nuvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"

def clear_all():
    """
    Очистка всех данных.
    """
    try:
        response = requests.delete(f"{API_URL}/clear")
        if response.status_code == 200:
            return " Все данные очищены"
        else:
            return f" Ошибка: {response.text}"
    except Exception as e:
        return f" Ошибка: {str(e)}"

def api_query(question):
    """
    Тестирование API.
    """
    if not question:
        return {"error": "Введите вопрос"}
    try:
        response = requests.post(f"{API_URL}/query", json={"question": question, "top_k": 5})
        return response.json() if response.status_code == 200 else {"error": response.text}
    except Exception as e:
        return {"error": str(e)}

def clear_chat():
    """
    Очистка чата.
    """
    return []

# Создание интерфейса Gradio

# CSS стили
css = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}
"""

# Создаем интерфейс
with gr.Blocks(title="Smart RAG System") as demo:
    
    # Заголовок
    gr.Markdown("""
    # Smart RAG System
    ### Гибридный поиск по документам (BM25 + Векторный поиск)
    Загрузите PDF и задавайте вопросы по его содержанию!
    """)
    
    # Вкладки
    with gr.Tabs():
                
        #  1: Чат
        
        with gr.TabItem(" Чат с документами"):
            with gr.Row():
                with gr.Column(scale=1):
                    # Загрузка файла
                    gr.Markdown("### Загрузка документа")
                    file_input = gr.File(
                        label="Выберите PDF файл",
                        file_types=[".pdf"]
                    )
                    # Используем upload_file_with_progress для красивого прогресса
                    upload_btn = gr.Button(" Загрузить и индексировать", variant="primary")
                    status_output = gr.Textbox(
                        label="Статус загрузки",
                        lines=4,
                        interactive=False
                    )
                    
                    # Статус сервера
                    gr.Markdown("### Статус системы")
                    status_check_btn = gr.Button(" Проверить статус")
                    status_info = gr.Textbox(
                        label="Информация",
                        lines=5,
                        interactive=False
                    )
                    
                    # Очистка
                    clear_btn = gr.Button(" Очистить все данные", variant="stop")
                    clear_output = gr.Textbox(
                        label="Результат очистки",
                        lines=2,
                        interactive=False
                    )
                
                with gr.Column(scale=2):
                    # Чат
                    gr.Markdown("### Задайте вопрос")
                    chatbot = gr.Chatbot(
                        label="Диалог с документами",
                        height=500
                    )
                    question_input = gr.Textbox(
                        label="Ваш вопрос",
                        placeholder="Например: О чем этот документ?",
                        lines=2
                    )
                    with gr.Row():
                        submit_btn = gr.Button(" Отправить", variant="primary")
                        clear_chat_btn = gr.Button(" Очистить чат")        
        
        # 2: Тестирование API
        
        with gr.TabItem(" Тестирование API"):
            gr.Markdown("""
            ### Прямые запросы к API
            Отправьте запрос и увидите сырой JSON ответ.
            """)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Загрузить PDF")
                    api_file = gr.File(
                        label="Выберите PDF",
                        file_types=[".pdf"]
                    )
                    api_upload_btn = gr.Button(" Загрузить")
                    api_upload_result = gr.Textbox(
                        label="Ответ API",
                        lines=6,
                        interactive=False
                    )
                
                with gr.Column():
                    gr.Markdown("#### Задать вопрос")
                    api_question = gr.Textbox(
                        label="Вопрос",
                        placeholder="Введите вопрос..."
                    )
                    api_query_btn = gr.Button(" Отправить")
                    api_query_result = gr.JSON(
                        label="Ответ API (JSON)"
                    )        
        
        # 3: Информация о проекте
        
        with gr.TabItem(" О проекте"):
            gr.Markdown("""
            ## Smart RAG System
            
            ### Что это?
            Система для умного поиска по документам с использованием гибридного подхода.
            
            ### Как это работает?
            1. **Загрузка PDF** → текст извлекается со всех страниц
            2. **Чанкинг** → текст разбивается на смысловые фрагменты
            3. **Индексация** → каждый чанк индексируется двумя способами:
               - **BM25** — поиск по ключевым словам
               - **Векторный поиск** (FAISS) — поиск по смыслу
            4. **Поиск** → при вопросе ищутся релевантные чанки
            5. **Ответ** → показываются найденные фрагменты с источниками
            
            ### Технологии
            - **Backend**: FastAPI, Python
            - **Поиск**: BM25 + FAISS
            - **Эмбеддинги**: SentenceTransformers (multilingual-e5-large)
            - **Frontend**: Gradio 6
            
            ### Ссылки
            - 📄 [Swagger API](http://localhost:8000/docs)
            """)   
    
    # Обработчики событий
    
    
    # Загрузка файла (используем upload_file_with_progress)
    upload_btn.click(
        fn=upload_file_with_progress,
        inputs=[file_input],
        outputs=[status_output]
    )
    
    # Отправка вопроса
    submit_btn.click(
        fn=ask_question,
        inputs=[question_input, chatbot],
        outputs=[chatbot]
    )
    
    # Отправка по Enter
    question_input.submit(
        fn=ask_question,
        inputs=[question_input, chatbot],
        outputs=[chatbot]
    )
    
    # Проверка статуса
    status_check_btn.click(
        fn=check_status,
        inputs=[],
        outputs=[status_info]
    )
    
    # Очистка данных
    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[clear_output]
    )
    
    # Очистка чата
    clear_chat_btn.click(
        fn=clear_chat,
        inputs=[],
        outputs=[chatbot]
    )
    
    # API тестирование: загрузка
    api_upload_btn.click(
        fn=upload_file,
        inputs=[api_file],
        outputs=[api_upload_result]
    )
    
    # API тестирование: запрос
    api_query_btn.click(
        fn=api_query,
        inputs=[api_question],
        outputs=[api_query_result]
    )
    
# Запуск


if __name__ == "__main__":
    print(" Запуск Gradio интерфейса...")
    print(" Подключение к API: http://localhost:8000")
    print(" Откройте в браузере: http://localhost:7860")
    print(" Убедитесь, что API сервер запущен!")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        css=css
    )