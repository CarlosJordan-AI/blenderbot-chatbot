from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr

# ─── 1. Load Model & Tokenizer ───────────────────────────────────────────────
model_name = "facebook/blenderbot-400M-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ─── 2. Conversation History ─────────────────────────────────────────────────
conversation_history = []

# ─── 3. Chat Function ────────────────────────────────────────────────────────
def chat(user_message, history):
    conversation_history.append(f"User: {user_message}")
    context = "\n".join(conversation_history[-10:])
    
    # ✅ Fixed: use tokenizer() directly instead of encode_plus()
    inputs = tokenizer(
        context,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
        repetition_penalty=1.3,
        do_sample=False
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    conversation_history.append(f"Bot: {response}")
    
    return response

# ─── 4. Gradio Interface ─────────────────────────────────────────────────────
iface = gr.ChatInterface(
    fn=chat,
    title="BlenderBot Chatbot",
    description="Conversational AI powered by Facebook's BlenderBot",
    examples=[
        "What is your favorite food?",
        "Tell me about yourself",
        "What do you think about AI?"
    ]
)

iface.launch()