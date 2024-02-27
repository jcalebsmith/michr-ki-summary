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

    st.title("Key Information Summary Generator for Informed Consent Documents")
    with st.expander("### Instructions for Using the Prototype GPT-4 Key Information Generator"):
        st.markdown("""
#### Purpose
This prototype is exclusively designed for generating **Key Information (KI) sections** of informed consent documents related to **biomedical research**. It is important to note that it has not been developed or tested for use with social/behavioral informed consents. The KI output from this prototype is considered a **draft**, and the **Principal Investigator (PI)/study team** bears full responsibility for evaluating its content and accuracy before submission to the Institutional Review Board (IRB).

#### Input of Information

- **Documents Eligibility**: Only input informed consent documents where **all sections**, except for the KI section, are fully drafted. The prototype's effectiveness depends on the complete and accurate information from other sections of the consent document to construct the KI section.
- **Accepted Formats**: The prototype accepts inputs in PDF, Word, or text file formats. The output will be provided in text format, which then must be **copied into the informed consent document, reviewed, and edited**. Please note that the output is **not saved** within the prototype.

#### Restrictions

It is crucial to avoid entering the following materials into the prototype due to their potentially sensitive nature. Doing so might violate the terms and conditions of a sponsor’s grant or contract:

- The investigational protocol
- Investigational drug or device brochure
- Any other information prohibited by the grant or contract
""")

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
