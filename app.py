import gradio as gr
import requests
import os
from typing import List, Dict, Optional
import json
import tempfile

# FastAPI endpoint URLs
BASE_URL = "https://512e-34-16-184-229.ngrok-free.app" 

def get_vector_db_info():
    """Get information about the vector database."""
    try:
        response = requests.get(f"{BASE_URL}/vector-db/info")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}

def upload_file(file):
    """Upload a file to the vector database."""
    try:
        if file is None:
            return {"status": "error", "message": "No file selected"}
            
        # Get the file path from Gradio's file object
        file_path = file.name
        
        # Open and read the file
        with open(file_path, 'rb') as f:
            files = {"file": (os.path.basename(file_path), f, "application/octet-stream")}
            response = requests.post(
                f"{BASE_URL}/vector-db/add-document",
                files=files
            )
            response.raise_for_status()
            result = response.json()
            
        return result
            
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"Network error: {str(e)}"}
    except Exception as e:
        return {"status": "error", "message": f"Upload error: {str(e)}"}

def delete_file(file_id: str):
    """Delete a file from the vector database."""
    try:
        response = requests.delete(f"{BASE_URL}/vector-db/delete-document/{file_id}")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}

def chat_with_documents(message: str, history: List[List[str]]):
    """Chat with the RAG-powered model."""
    try:
        # Convert history to the format expected by the API
        api_history = []
        for human, ai in history:
            api_history.append({"role": "user", "content": human})
            api_history.append({"role": "assistant", "content": ai})

        # Make API request
        response = requests.post(
            f"{BASE_URL}/chat",
            json={"query": message, "history": api_history}
        )
        response.raise_for_status()
        result = response.json()

        # Format the response with sources
        answer = result["answer"]
        sources = result["sources"]
        if sources:
            answer += "\n\nSources:\n" + "\n".join([f"- {s['filename']}" for s in sources])

        # Return the message in the format expected by Gradio Chatbot
        # The history will be updated with the new message pair
        history.append([message, answer])
        return history
    except Exception as e:
        error_message = f"Error: {str(e)}"
        history.append([message, error_message])
        return history

def get_uploaded_files_info():
    """Get list of files (filename and file_id) in the vector database."""
    try:
        response = requests.get(f"{BASE_URL}/vector-db/info")
        response.raise_for_status()
        info = response.json()
        print("[DEBUG] /vector-db/info response:", info)
        return info.get("files", [])
    except Exception as e:
        print(f"[DEBUG] Error fetching files: {e}")
        return []

def delete_files_from_db(filenames: list):
    """Delete multiple files from the vector database by filename."""
    if not filenames:
        return "No files selected."
    files_info = get_uploaded_files_info()
    deleted = []
    errors = []
    for fname in filenames:
        file_id = None
        for f in files_info:
            if f["filename"] == fname:
                file_id = f["file_id"]
                break
        if file_id:
            try:
                response = requests.delete(f"{BASE_URL}/vector-db/delete-document/{file_id}")
                response.raise_for_status()
                deleted.append(fname)
            except Exception as e:
                errors.append(f"{fname}: {str(e)}")
        else:
            errors.append(f"{fname}: Not found")
    msg = ""
    if deleted:
        msg += f"Deleted: {', '.join(deleted)}. "
    if errors:
        msg += f"Errors: {'; '.join(errors)}"
    return msg or "No files deleted."

def get_filenames_list():
    files_info = get_uploaded_files_info()
    return [f["filename"] for f in files_info]

# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# NUST Bank Financial QnA Chatbot")
    
    with gr.Tab("File Management"):
        gr.Markdown("Upload and manage your financial documents here.")
        
        with gr.Row():
            with gr.Column(scale=2):
                file_upload = gr.File(
                    label="Upload Financial Document",
                    file_types=[".pdf", ".docx", ".txt", ".csv", ".xlsx", ".json"],
                    file_count="single"
                )
                upload_btn = gr.Button("Upload")
                status_box = gr.Textbox(label="Status", interactive=False)
            
            with gr.Column(scale=1):
                gr.Markdown("### Vector Database Status")
                db_info = gr.JSON(label="Database Info")
                refresh_btn = gr.Button("Refresh Status")
                
                gr.Markdown("### Delete Files")
                file_checkboxes = gr.CheckboxGroup(
                    label="Select financial files to delete",
                    choices=get_filenames_list(),
                    interactive=True
                )
                delete_btn = gr.Button("Delete Selected Files", variant="stop")
                delete_status = gr.Textbox(label="Delete Status", interactive=False)
                # Debug: Show raw file list
                file_list_debug = gr.JSON(label="[DEBUG] Raw File List from API")
        
        # Define button actions
        def update_file_checkboxes():
            files = get_filenames_list()
            return gr.CheckboxGroup.update(choices=files, value=[])
        def update_file_list_debug():
            return get_uploaded_files_info()
        
        upload_btn.click(
            fn=upload_file,
            inputs=[file_upload],
            outputs=[status_box]
        ).then(
            fn=get_vector_db_info,
            inputs=None,
            outputs=[db_info]
        ).then(
            fn=update_file_checkboxes,
            inputs=None,
            outputs=[file_checkboxes]
        ).then(
            fn=update_file_list_debug,
            inputs=None,
            outputs=[file_list_debug]
        )
        
        refresh_btn.click(
            fn=get_vector_db_info,
            inputs=None,
            outputs=[db_info]
        ).then(
            fn=update_file_checkboxes,
            inputs=None,
            outputs=[file_checkboxes]
        ).then(
            fn=update_file_list_debug,
            inputs=None,
            outputs=[file_list_debug]
        )
        
        delete_btn.click(
            fn=delete_files_from_db,
            inputs=[file_checkboxes],
            outputs=[delete_status]
        ).then(
            fn=get_vector_db_info,
            inputs=None,
            outputs=[db_info]
        ).then(
            fn=update_file_checkboxes,
            inputs=None,
            outputs=[file_checkboxes]
        ).then(
            fn=update_file_list_debug,
            inputs=None,
            outputs=[file_list_debug]
        )
    
    with gr.Tab("Chat"):
        gr.Markdown("Ask financial questions about your documents using the NUST Bank Financial QnA Chatbot.")
        
        chatbot = gr.Chatbot(
            label="Financial QnA Chat History",
            height=500,
            show_copy_button=True
        )
        
        with gr.Row():
            msg = gr.Textbox(
                label="Message",
                placeholder="Ask a financial question about your documents...",
                lines=1,
                show_label=False
            )
            submit_btn = gr.Button("Send", variant="primary")
        
        # Define chat actions
        submit_btn.click(
            fn=chat_with_documents,
            inputs=[msg, chatbot],
            outputs=[chatbot]
        ).then(
            fn=lambda: "",
            inputs=None,
            outputs=[msg]
        )
        
        msg.submit(
            fn=chat_with_documents,
            inputs=[msg, chatbot],
            outputs=[chatbot]
        ).then(
            fn=lambda: "",
            inputs=None,
            outputs=[msg]
        )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True) 