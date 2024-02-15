import streamlit as st
import os
import base64
from generate_ki_summary import generate_summary

st.set_page_config(page_title="Informed Consent Summary Generator", layout="wide")

# Constants
SUPPORTED_FILETYPES = {'.pdf', '.docx'}
UPLOADS_DIR = "uploads"

def is_supported_filetype(filename: str) -> bool:
    """Check if the file type is supported."""
    return os.path.splitext(filename)[1] in SUPPORTED_FILETYPES

def execute_script(document_path: str) -> str:
    """Call the generate_summary function with the document path."""
    return generate_summary(document_path)

def get_image_base64(path: str) -> str:
    with open(path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return f"data:image/png;base64,{encoded_string}"

def create_header():
    left_logo_base64 = get_image_base64('./images/logo_ethicscompliance.png')
    right_logo_base64 = get_image_base64('./images/michr-hd-logo.png')
    
    header_html = f"""
        <div style="background-color:#00274C; padding: 10px; min-height: 60px;">
            <img src="{left_logo_base64}" style="float:left;height:40px;">
            <img src="{right_logo_base64}" style="float:right;height:40px;">
        </div>
        """
    st.markdown(header_html, unsafe_allow_html=True)

def main():
    create_header()

    st.title("Generate Key Information Summary")

    col1, col2 = st.columns([1, 3])
    with col1:
        uploaded_file = st.file_uploader("Upload Informed Consent", type=list(SUPPORTED_FILETYPES), accept_multiple_files=False)

    if uploaded_file is not None:
        if not is_supported_filetype(uploaded_file.name):
            st.error("File type not supported.")
        else:
            st.success(f"File '{uploaded_file.name}' uploaded successfully!")

            if st.button("Generate Key Information Summary"):
                with st.spinner("Generating the Key Information section. This may take a minute..."):
                    if not os.path.exists(UPLOADS_DIR):
                        os.makedirs(UPLOADS_DIR)
                    document_path = os.path.join(UPLOADS_DIR, uploaded_file.name)
                    with open(document_path, "wb") as file:
                        file.write(uploaded_file.getbuffer())

                    response = execute_script(document_path)

                    # Display the response in a text area
                    st.title("Key Information")
                    st.text_area(label="Key Information Summary", label_visibility="hidden",value=response, height=500, disabled=False)
                    
                    # Provide a download button for the summary
                    st.download_button("Download Summary", data=response, file_name="summary.txt", mime="text/plain")

if __name__ == "__main__":
    main()
