import streamlit as st
import os
import io
import base64
import fitz
import numpy as np
from PIL import Image
import PyPDF2
import tempfile
from keybert import KeyBERT
from docx import Document
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


# Load models once
keybert_model = KeyBERT(model=r"sentence_transformer\model")
embedding_model = SentenceTransformer(model_name_or_path=r"sentence_transformer\model")

# Initialize session state
if "text" not in st.session_state:
    st.session_state.text = ""

if "keywords" not in st.session_state:
    st.session_state.keywords = []

if "encoding" not in st.session_state:
    st.session_state.encoding= None




def convert_word_to_pdf(word_doc_io):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_word_file:
            temp_word_path = temp_word_file.name

        with open(temp_word_path, "wb") as word_writer:
            word_writer.write(io.BytesIO(word_doc_io).read())

        temp_pdf_path = temp_word_path.replace("docx", "pdf")   
        
        with open(temp_pdf_path, "rb") as pdf_file:
            pdf_object = io.BytesIO(pdf_file.read())

        
    finally:
        if os.path.exists(temp_word_path):
            os.remove(temp_word_path)
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
        return pdf_object

def reset_keywords():
    st.session_state.keywords = []
    st.session_state.text = ""

def annotate_pdf(pdf_document):
    
    # page containing word
    page_containing = []
    # iterate through pages.
    for page_num in range(pdf_document.page_count):
        # load the page
        page = pdf_document.load_page(page_num)
        # search for text
        text_instances = page.search_for(st.session_state.keywords[max_index])
        if len(text_instances) > 0:
            page_containing.append(page_num+1)
        for inst in text_instances:
            highlight = page.add_highlight_annot(inst)
            highlight.set_colors({"stroke": (1, 1, 0)})  # Yellow highlight
            highlight.update()
        
    # convert to btyes
    pdf_bytes = pdf_document.tobytes()
    pdf_document.close()

    # encode the data
    pdf_data = base64.b64encode(pdf_bytes).decode("utf-8")
    # return page containing
    return pdf_data, page_containing

st.header("Smart Finder: Find Contextual Words or Phrases in PDF or Word Document.")
image = Image.open("background_image.jpg")
image_array = np.array(image)
st.image(image_array)
# File uploader
file = st.file_uploader(label="Upload File", on_change=reset_keywords, accept_multiple_files=False)

if file:
    # Extract file extension and process PDF
    filename, file_extension = os.path.splitext(file.name)
    try:
        if file_extension == ".pdf":
            pdf_reader = PyPDF2.PdfReader(file)
            # Combine text from all pages
            full_text = " ".join([page.extract_text() for page in pdf_reader.pages])
            # Extract keywords only if they haven't been calculated yet
            if not st.session_state.keywords:
                with st.spinner("Preprocessing Document..."):
                    keywords = keybert_model.extract_keywords(full_text, top_n=-1)
                    st.session_state.keywords = [k[0] for k in keywords]
                    st.session_state.encoding = embedding_model.encode(st.session_state.keywords)
        elif file_extension == ".docx" or file_extension==".doc":
            doc = Document(file)
            full_text = " ".join([paragraph.text for paragraph in doc.paragraphs])
            if not st.session_state.keywords:
                with st.spinner("Processing Document..."):
                    keywords = keybert_model.extract_keywords(full_text, top_n=-1)
                    st.session_state.keywords = [k[0] for k in keywords]
                    st.session_state.encoding = embedding_model.encode(st.session_state.keywords)
        else:
            raise ValueError(f"{file_extension} is not supported. Upload either Pdf or Word Document.")
    except Exception as e:
        st.error(f"An Error Occcured: ({e})")

# Text input for word or phrase
word_or_phrase = st.text_input("Enter Word or Phrase to Find:", key="text")

max_index = 0
# Check if keywords are ready before processing
if st.session_state.keywords and word_or_phrase and st.session_state.encoding is not None:
    # Compute embeddings
    word_or_phrase_embedding = embedding_model.encode(word_or_phrase)
    # Compute cosine similarity
    similarity = cos_sim(word_or_phrase_embedding, st.session_state.encoding)

    # Get most similar keyword
    if similarity.max() > 0.7:
        max_index = similarity.argmax()
        st.info(f"Most similar keyword to {word_or_phrase} is: **'{st.session_state.keywords[max_index]}'** with above 70% similarity score.")
    else:
        st.warning("The document does not contain words that are 70% above similar.")    


    if max_index:
        if os.path.splitext(file.name)[1] == ".pdf":
            pdf_document = fitz.open(stream=file.getvalue())
            pdf_data, page_containing = annotate_pdf(pdf_document)
        elif os.path.splitext(file.name)[1] == ".docx" or os.path.splitext(file.name)[1] == ".doc":
            word_document = convert_word_to_pdf(file)
            pdf_document = fitz.open(stream=word_document)
            pdf_data, page_containing = annotate_pdf(pdf_document)
            
        st.success(f"The word '{st.session_state.keywords[max_index]}' occur(s) on pages {', '.join(map(str, page_containing))}. You can scroll to these pages.")

        # Embedding PDF in HTML
        pdf_display = F'<iframe src="data:application/pdf;base64,{pdf_data}#page={page_containing[0]}&zoom=90%" width="100%" height="1000" type="application/pdf"></iframe>'
        # Displaying File
        st.markdown(pdf_display, unsafe_allow_html=True)

    