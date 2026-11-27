
import streamlit as st
import base64
import os
from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
from PIL import Image


# ========== SETUP ==========

st.set_page_config(
    page_title="PaperMind AI",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "Data" / "images"

# Separate models for text and vision
TEXT_MODEL = "openai/gpt-oss-20b"
VISION_MODEL = "qwen/qwen3.8-27b"


# ========== LOAD MODELS ==========

@st.cache_resource
def load_models():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2",
        device="cpu"
    )

    clip_model = SentenceTransformer(
        "clip-ViT-B-32",
        device="cpu"
    )

    chroma_client = chromadb.PersistentClient(
        path=str(BASE_DIR / "chroma_db")
    )

    collection = chroma_client.get_collection(
        name="paper_30"
    )

    img_collection = chroma_client.get_collection(
        name="paper_30_images"
    )

    groq_client = Groq(
        api_key=st.secrets["GROQ_API_KEY"]
    )

    return (
        model,
        clip_model,
        collection,
        img_collection,
        groq_client
    )


model, clip_model, collection, img_collection, groq_client = load_models()


# ========== CHAT HISTORY ==========

if "chats" not in st.session_state:

    st.session_state.chats = {
        "What is PMD?": [
            {
                "role": "user",
                "content": "What is PMD?"
            },
            {
                "role": "assistant",
                "content": "PMD is Posterior Matching Distillation..."
            }
        ]
    }

if "current_chat" not in st.session_state:
    st.session_state.current_chat = "What is PMD?"


# ========== SIDEBAR ==========

with st.sidebar:

    st.title("📚 PaperMind AI")

    st.text_input(
        "🔍 Search for chats",
        placeholder="Search for chats"
    )

    if st.button(
        "➕ New conversation",
        use_container_width=True
    ):

        new_title = f"Chat {len(st.session_state.chats) + 1}"

        st.session_state.chats[new_title] = []

        st.session_state.current_chat = new_title

        st.rerun()

    st.divider()

    st.write("**TODAY**")

    for title in list(st.session_state.chats.keys()):

        if st.button(
            f"💬 {title}",
            key=f"chat_{title}",
            use_container_width=True
        ):

            st.session_state.current_chat = title
            st.rerun()

    st.divider()

    st.write("**Settings**")

    paper = st.selectbox(
        "Choose Paper",
        [
            "All Papers",
            "01_2611.26629v1",
            "11_2611.26607v1"
        ]
    )


# ========== MAIN CHAT AREA ==========

st.header(
    f"🤖 PaperMind Bot - {st.session_state.current_chat}"
)

current_messages = st.session_state.chats[
    st.session_state.current_chat
]

for msg in current_messages:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])


query = st.chat_input(
    "Ask any question about your document..."
)


# ========== QUESTION PROCESSING ==========

if query:

    st.session_state.chats[
        st.session_state.current_chat
    ].append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.write(query)

    # Router: Text or Image?

    image_keywords = [
        "figure",
        "graph",
        "diagram",
        "image",
        "chart",
        "photo",
        "picture",
        "plot"
    ]

    is_image = any(
        word in query.lower()
        for word in image_keywords
    )

    # =====================================
    # PART 1: TEXT RAG
    # =====================================

    if not is_image:

        with st.spinner("Searching paper context..."):

            q_emb = model.encode([query]).tolist()

            # Paper filter
            where_filter = None

            if paper != "All Papers":
                where_filter = {"source": paper}

            res = collection.query(
                query_embeddings=q_emb,
                n_results=5,
                where=where_filter
            )

            documents = res["documents"][0]

            metadatas = res["metadatas"][0]

            if not documents:

                answer = "No relevant context found in the selected paper."

            else:

                # Convert retrieved documents into string
                context = "\n\n".join(documents[:3])

                # Prepare source information
                sources = "\n".join(
                    [
                        f"Source: {m.get('source', 'Unknown')}, "
                        f"Page: {m.get('page', 'Unknown')}"
                        for m in metadatas[:3]
                    ]
                )

                # Groq TEXT model
                response = groq_client.chat.completions.create(

                    model=TEXT_MODEL,

                    messages=[

                        {
                            "role": "system",
                            "content": (
                                "You are PaperMind AI. "
                                "Answer only using the provided context. "
                                "If the answer is not present, say so. "
                                "Include the relevant source."
                            )
                        },

                        {
                            "role": "user",
                            "content": (
                                f"Context:\n{context}\n\n"
                                f"Question: {query}\n\n"
                                f"Sources:\n{sources}"
                            )
                        }

                    ],

                    temperature=0.2
                )

                answer = response.choices[0].message.content

        with st.chat_message("assistant"):

            st.write(answer)

        st.session_state.chats[
            st.session_state.current_chat
        ].append({
            "role": "assistant",
            "content": answer
        })


    # =====================================
    # PART 2: IMAGE RAG + VISION
    # =====================================

    else:

        with st.spinner("Searching relevant paper images..."):

            q_emb = clip_model.encode(
                [query]
            ).tolist()

            where_filter = None

            if paper != "All Papers":
                where_filter = {"source": paper}

            res = img_collection.query(

                query_embeddings=q_emb,

                n_results=3,

                where=where_filter

            )

            files = res["documents"][0]

            metas = res["metadatas"][0]

        if not files:

            st.warning("No relevant images found.")

        else:

            st.write("### 🖼️ Found Images")

            valid_images = []

            cols = st.columns(3)

            # Display retrieved images

            for i, (filename, metadata) in enumerate(
                zip(files, metas)
            ):

                img_path = IMAGE_DIR / os.path.basename(filename)

                with cols[i]:

                    if img_path.exists():

                        st.image(
                            str(img_path),
                            caption=(
                                f"{metadata.get('source', 'Unknown')} "
                                f"- Page {metadata.get('page', 'Unknown')}"
                            ),
                            use_container_width=True
                        )

                        valid_images.append(
                            (img_path, metadata)
                        )

                    else:

                        st.error(
                            f"Image not found: {img_path}"
                        )

            # Analyze the first valid retrieved image

            if valid_images:

                img_path, metadata = valid_images[0]

                with st.spinner("Analyzing image using Vision AI..."):

                    # Read image
                    with open(img_path, "rb") as image_file:

                        img_bytes = image_file.read()

                    img_base64 = base64.b64encode(
                        img_bytes
                    ).decode("utf-8")

                    # Detect image format
                    image_format = Image.open(
                        img_path
                    ).format.lower()

                    mime_type = {
                        "jpeg": "image/jpeg",
                        "jpg": "image/jpeg",
                        "png": "image/png",
                        "webp": "image/webp"
                    }.get(image_format, "image/png")

                    # Groq VISION model
                    response = groq_client.chat.completions.create(

                        model=VISION_MODEL,

                        messages=[

                            {
                                "role": "user",

                                "content": [

                                    {
                                        "type": "text",

                                        "text": (
                                            "Analyze this research paper figure carefully. "
                                            "Read the title, labels, axes and legend exactly. "
                                            "Do not reverse or assume the legend meanings. "
                                            "Explain the figure and answer this question: "
                                            f"{query}"
                                        )
                                    },

                                    {
                                        "type": "image_url",

                                        "image_url": {
                                            "url": (
                                                f"data:{mime_type};base64,"
                                                f"{img_base64}"
                                            )
                                        }
                                    }

                                ]
                            }

                        ],

                        temperature=0.2

                    )

                    answer = response.choices[0].message.content

                with st.chat_message("assistant"):

                    st.write(answer)

                    st.info(
                        f"Source: {metadata.get('source', 'Unknown')} "
                        f"| Page: {metadata.get('page', 'Unknown')} "
                        f"| Image: {img_path.name}"
                    )

                st.session_state.chats[
                    st.session_state.current_chat
                ].append({
                    "role": "assistant",
                    "content": answer
                })
