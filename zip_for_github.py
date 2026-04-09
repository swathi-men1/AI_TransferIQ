import os
import zipfile

def create_github_release():
    zip_filename = "TransferIQ_GitHub_Ready.zip"
    
    # The files that make up the perfect portfolio project
    files_to_include = [
        "transferiq_website.html",         # The Application
        "complete_integrated_dataset.csv", # The Data
        "README.md",                       # Documentation
        "Model_Evaluation_Report.md",      # Report
        "DATASETS_SUMMARY.md",             # Report
        "TransferIQ_Demo.pptx",            # Presentation
    ]
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files_to_include:
            if os.path.exists(file):
                zipf.write(file)
            else:
                print(f"Warning: {file} not found.")
                
    print(f"Successfully created {zip_filename}!")

if __name__ == "__main__":
    create_github_release()
