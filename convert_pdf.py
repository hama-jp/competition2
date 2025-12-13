import os
import markdown
from xhtml2pdf import pisa

# Path
BASE_DIR = 'experiments/exp48_clustering_pca'
MD_FILE = os.path.join(BASE_DIR, 'solution_description.md')
PDF_FILE = os.path.join(BASE_DIR, 'solution_description.pdf')
FONT_PATH = r"C:\Windows\Fonts\msmincho.ttc"

def convert_md_to_pdf(source_md, output_pdf):
    # 1. Read Markdown
    with open(source_md, 'r', encoding='utf-8') as f:
        text = f.read()

    # 2. Convert to HTML
    html_body = markdown.markdown(text, extensions=['tables'])

    # 3. Create HTML Wrapper with Japanese Font
    # MS Mincho is standard on Windows.
    html_content = f"""
    <html>
    <head>
        <style>
            @font-face {{
                font-family: "JP";
                src: url("{FONT_PATH}");
            }}
            body {{
                font-family: "JP", sans-serif;
                font-size: 12pt;
                line-height: 1.5;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            code {{
                background-color: #f4f4f4;
                padding: 2px 4px;
                border-radius: 4px;
                font-family: "Courier New", monospace;
            }}
            pre {{
                background-color: #f8f8f8;
                padding: 10px;
                border: 1px solid #ddd;
                white-space: pre-wrap;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
        </style>
    </head>
    <body>
        {html_body}
    </body>
    </html>
    """

    # 4. Write PDF
    with open(output_pdf, "w+b") as result_file:
        pisa_status = pisa.CreatePDF(
            html_content,
            dest=result_file,
            encoding='utf-8'
        )

    if pisa_status.err:
        print(f"Error converting to PDF: {pisa_status.err}")
        return False
    return True

if __name__ == "__main__":
    print(f"Converting {MD_FILE} to PDF...")
    if os.path.exists(MD_FILE):
        success = convert_md_to_pdf(MD_FILE, PDF_FILE)
        if success:
            print(f"Successfully created: {PDF_FILE}")
        else:
            print("Failed.")
    else:
        print(f"File not found: {MD_FILE}")
