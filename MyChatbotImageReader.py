import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from PIL import Image
import base64
import io

st.set_page_config(page_title="Gemma Vision Chatbot", layout="wide")
st.title("🖼️ Gemma Vision Chatbot")
st.caption("Upload an image and ask questions about it")

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("Upload an image and ask me anything about it!")

# Init LLM - Gemma 3 4b with vision
llm = ChatOllama(model="gemma3:4b", temperature=0.3)

# Sidebar for image upload
with st.sidebar:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
    if st.button("Clear Chat"):
        msgs.clear()
        st.rerun()

# Display chat history
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)


# Helper: convert PIL image to base64
def image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


# Chat input
if prompt := st.chat_input("Ask about the image..."):
    st.chat_message("user").write(prompt)

    # Build multimodal message
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        # Show image in chat
        st.chat_message("user").image(image, width=300)

        img_b64 = image_to_base64(image)
        human_message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": f"data:image/png;base64,{img_b64}"}
            ]
        )
    else:
        st.warning("Please upload an image first.")
        st.stop()

    # Get response from Gemma
    with st.chat_message("assistant"):
        with st.spinner("Gemma is looking..."):
            # Include chat history for context
            history = msgs.messages
            response = llm.invoke([*history, human_message])
            st.write(response.content)

    # Save to memory
    msgs.add_user_message(prompt)
    msgs.add_ai_message(response.content)