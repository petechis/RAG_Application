import os 
import tempfile
import streamlit as st

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import PyPDFLoader

from langchain_community.llms import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

#import PyPDF2

def clear_history():
    st.session_state.my_text = ""

def read_file(file_path):
    # Get the file extension
    _, file_extension = os.path.splitext(file_path)
    
    if file_extension == '.pdf':
        return read_pdf(file_path)
    elif file_extension == '.docx':
        return read_docx(file_path)
    elif file_extension == '.txt':
        return read_txt(file_path)
    else:
        raise ValueError("Unsupported file format")

def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
        return text

def read_docx(file_path):
    doc = docx.Document(file_path)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

#Read files and return the contents in a list 
def process_docs(uploads):
    documents = []
    for file in uploads:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.read())
            filename = file.name
            if filename.endswith('.pdf'):
                loader = PyPDFLoader(tmp_file.name)
                documents.extend(loader.load())
            elif filename.endswith('.docx') or filename.endswith('.doc'):
                loader = Docx2txtLoader(tmp_file.name)
                documents.extend(loader.load())
            elif filename.endswith('.txt'):
                loader = TextLoader(tmp_file.name)
                documents.extend(loader.load())
    return documents

def ask_and_get_answer(vector_store, q, k=3): 
	from langchain.chains import RetrievalQA 
	from langchain.chat_models import ChatOpenAI 
 
	llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1) 
	retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k}) 
	chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever) 
 
	answer = chain.run(q) 
	return answer 

st.image('img/openai_logo_original.png')
st.subheader('Simulated RAG and LLM Question/Answer Application.')

with st.sidebar:
    api_key = st.text_input('OpenAI API Key', type='password')

    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key
    upload_file = st.file_uploader('Upload a file to add to llm-info.', type=['pdf','docx','txt'])

    chunk_size = st.number_input('Chunk Size: ', min_value=100, max_value=2048, on_change=clear_history)

    k=st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)

    add_data = st.button('Add Data', on_click=clear_history)
    st.write()

def initialize_vectorstore(data, session_state):
    # Initialize the embeddings model
    embeddings = OpenAIEmbeddings()    

    # Convert data to embeddings
   
    data_embeddings = [embeddings.embed_documents(text) for text in data]   

    # Create a FAISS index
    dimension = len(data_embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(data_embeddings))

    # Create a vector store
    vectorstore = FAISS(index, embeddings)

    # Store the vector store in the session state
    session_state['vs'] = vectorstore

    return vectorstore 

if upload_file and add_data:
    with st.spinner('Reading, Chunking and embedding file ...'):
        bytes_data= upload_file.read()
        file_name = os.path.join('./', upload_file.name)
        file_data = None

        with open(file_name, 'wb') as f:
            f.write(bytes_data)            

        data = read_file(file_name)  
        session_state = {} 
        initialize_vectorstore(data, session_state)

q = st.text_input("Ask a question about the uploaded files with current augemented context  (e.g. Company info)")

if q:
    print('Question asked --------------------------')
    if 'vs' in st.session_state:
        vector_store = st.session_state.vs
        st.write(f'k: {k}')

        answer= ask_and_get_answer(vector_store, q, k)
        print('Answer prepared asked --------------------------')
        st.text_area('LLM Answer', value=answer)

        st.divider()

        if 'history' not in st.session_state:
            st.session_state.history =''

