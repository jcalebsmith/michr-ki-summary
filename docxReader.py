"""
Original source code from: https://github.com/run-llama/llama-hub/tree/c8076387f3249dc7b2d410b968d9f495d4beb02d/llama_hub/file/docx
commit: a8c11d9
Manually reproducing the base.py file here to avoid Streamlit directory permission issues.
"""

"""Read Microsoft Word files."""

from pathlib import Path
from typing import Dict, List, Optional

from llama_index.readers.base import BaseReader
from llama_index.readers.schema.base import Document


class DocxReader(BaseReader):
    """Docx Reader."""

    def load_data(
        self, file: Path, extra_info: Optional[Dict] = None
    ) -> List[Document]:
        """Parse file."""
        import docx2txt

        text = docx2txt.process(file)
        metadata = {"file_name": file.name}

        if extra_info is not None:
            metadata.update(extra_info)

        return [Document(text=text, extra_info=metadata)]