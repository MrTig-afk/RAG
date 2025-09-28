import os

def convert_pdf_files(folder_path):
    """Convert all _pdf files to .pdf files in the specified folder."""
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf_'):
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, filename[:-4] + '.pdf')
            
            try:
                os.rename(old_path, new_path)
                print(f"Converted: {filename} → {filename[:-4]}.pdf")
            except Exception as e:
                print(f"Error converting {filename}: {e}")

# Usage - replace with your actual folder path
folder_path = r"D:\MSDS 2nd Sem\Case Studies in Data Science\Assignments\Github\pdfs"  # Windows
# folder_path = "/path/to/your/folder"   # Linux/Mac

convert_pdf_files(folder_path)
print("Conversion completed!")