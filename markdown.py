import os
import pymupdf4llm

def convert_pdf_to_markdown(pdf_path):
    """
    Converts a PDF file to Markdown format using pymupdf4llm.
    """
    try:
        # Convert the PDF to Markdown
        md_text = pymupdf4llm.to_markdown(pdf_path)
        return md_text
    except Exception as e:
        print(f"Error converting {pdf_path}: {e}")
        return None

def save_markdown(md_text, output_path):
    """
    Saves the Markdown text to a file.
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as md_file:
            md_file.write(md_text)
        print(f"Markdown saved to {output_path}")
    except Exception as e:
        print(f"Error saving {output_path}: {e}")

def process_pdfs_in_folder(folder_path):
    """
    Processes all .pdf_ files in the specified folder, converting them to Markdown.
    """
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf_'):
            pdf_path = os.path.join(folder_path, filename)
            md_text = convert_pdf_to_markdown(pdf_path)
            if md_text:
                # Remove the '.pdf_' extension and add '.md'
                md_filename = f"{os.path.splitext(filename)[0]}.md"
                md_path = os.path.join(folder_path, md_filename)
                save_markdown(md_text, md_path)

if __name__ == "__main__":
    folder_path = input("Enter the path to the folder containing .pdf_ files: ")
    process_pdfs_in_folder(folder_path)
