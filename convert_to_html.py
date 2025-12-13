import os
import markdown

# Path
BASE_DIR = 'experiments/exp48_clustering_pca'
MD_FILE = os.path.join(BASE_DIR, 'solution_description.md')
HTML_FILE = os.path.join(BASE_DIR, 'solution_description.html')

def convert_md_to_html(source_md, output_html):
    # 1. Read Markdown
    with open(source_md, 'r', encoding='utf-8') as f:
        text = f.read()

    # 2. Convert to HTML
    html_body = markdown.markdown(text, extensions=['tables'])

    # 3. Create HTML Wrapper (Github Style)
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <title>Solution Description</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
                font-size: 16px;
                line-height: 1.5;
                word-wrap: break-word;
                color: #24292e;
                margin: 0 auto;
                padding: 45px;
                max-width: 980px;
                border: 1px solid #ddd;
                background-color: #fff;
            }}
            h1, h2, h3 {{
                margin-top: 24px;
                margin-bottom: 16px;
                font-weight: 600;
                line-height: 1.25;
            }}
            h1 {{ font-size: 2em; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }}
            h2 {{ font-size: 1.5em; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }}
            h3 {{ font-size: 1.25em; }}
            code {{
                padding: 0.2em 0.4em;
                margin: 0;
                font-size: 85%;
                background-color: #f6f8fa;
                border-radius: 6px;
                font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
            }}
            pre {{
                padding: 16px;
                overflow: auto;
                font-size: 85%;
                line-height: 1.45;
                background-color: #f6f8fa;
                border-radius: 6px;
            }}
            table {{
                border-spacing: 0;
                border-collapse: collapse;
                margin-bottom: 16px;
                width: 100%;
            }}
            table th, table td {{
                padding: 6px 13px;
                border: 1px solid #dfe2e5;
            }}
            table th {{
                font-weight: 600;
                background-color: #f6f8fa;
            }}
            table tr:nth-child(2n) {{
                background-color: #f6f8fa;
            }}
            ul {{ padding-left: 2em; }}
        </style>
    </head>
    <body>
        {html_body}
    </body>
    </html>
    """

    # 4. Write HTML
    with open(output_html, "w", encoding='utf-8') as f:
        f.write(html_content)
    
    return True

if __name__ == "__main__":
    print(f"Converting {MD_FILE} to HTML...")
    if os.path.exists(MD_FILE):
        success = convert_md_to_html(MD_FILE, HTML_FILE)
        if success:
            print(f"Successfully created: {HTML_FILE}")
        else:
            print("Failed.")
    else:
        print(f"File not found: {MD_FILE}")
