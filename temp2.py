import fitz  # PyMuPDF
from PIL import Image
import io
import re
from IPython.display import display, Markdown
import ipywidgets as widgets

class PDFDocumentExtractor:
    def __init__(self):
        self.pdf_docs = {}
        self.current_doc = None
        self.current_page = 0
        self.search_results = []
        
    def load_pdf(self, file_path):
        """Load a PDF document"""
        try:
            doc = fitz.open(file_path)
            filename = file_path.split('/')[-1]
            self.pdf_docs[filename] = doc
            self.current_doc = doc
            return f"Successfully loaded {filename} with {len(doc)} pages"
        except Exception as e:
            return f"Error loading PDF: {str(e)}"
    
    def extract_page_content(self, page_num=None, show_images=True):
        """Extract and display content from a specific page"""
        if not self.current_doc:
            return "No document loaded"
            
        if page_num is None:
            page_num = self.current_page
            
        try:
            page = self.current_doc[page_num]
            output = []
            
            # Extract text blocks with formatting
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if block['type'] == 0:  # Text block
                    block_text = ""
                    for line in block["lines"]:
                        for span in line["spans"]:
                            # Preserve formatting
                            if span['flags'] & 2:  # Italic
                                block_text += f"*{span['text']}*"
                            elif span['flags'] & 8:  # Bold
                                block_text += f"**{span['text']}**"
                            else:
                                block_text += span['text']
                        block_text += "\n"
                    output.append(block_text)
                
                elif block['type'] == 1 and show_images:  # Image block
                    img = block['image']
                    if img:
                        try:
                            pil_img = Image.open(io.BytesIO(img))
                            display(pil_img)
                            output.append("\n[IMAGE EXTRACTED]\n")
                        except:
                            output.append("\n[IMAGE (failed to display)]\n")
            
            # Display the formatted content
            display(Markdown(f"## Page {page_num + 1} Content\n"))
            for content in output:
                display(Markdown(content))
            
            return f"Displayed content from page {page_num + 1}"
        except Exception as e:
            return f"Error extracting page content: {str(e)}"
    
    def search_content(self, query):
        """Search for specific content across the document"""
        if not self.current_doc:
            return "No document loaded"
            
        self.search_results = []
        results_text = []
        
        for page_num in range(len(self.current_doc)):
            page = self.current_doc[page_num]
            text = page.get_text()
            
            if query.lower() in text.lower():
                # Find context around the match
                matches = re.finditer(rf"(.{{0,30}}{re.escape(query)}.{{0,30}})", text, re.IGNORECASE)
                for match in matches:
                    self.search_results.append({
                        'page': page_num,
                        'context': match.group(1)
                    })
                    results_text.append(f"Page {page_num + 1}: ...{match.group(1)}...")
        
        if results_text:
            display(Markdown("### Search Results\n"))
            for i, result in enumerate(results_text, 1):
                display(Markdown(f"{i}. {result}"))
            return f"Found {len(results_text)} matches for '{query}'"
        else:
            return f"No matches found for '{query}'"
    
    def show_search_result(self, result_num):
        """Display a specific search result in context"""
        if not self.search_results or result_num < 1 or result_num > len(self.search_results):
            return "Invalid result number"
            
        result = self.search_results[result_num - 1]
        self.current_page = result['page']
        return self.extract_page_content(result['page'])
    
    def get_diagrams(self, page_num=None):
        """Extract and display all diagrams from a page"""
        if not self.current_doc:
            return "No document loaded"
            
        if page_num is None:
            page_num = self.current_page
            
        try:
            page = self.current_doc[page_num]
            images = []
            
            for img in page.get_images():
                xref = img[0]
                base_image = self.current_doc.extract_image(xref)
                image_bytes = base_image["image"]
                pil_img = Image.open(io.BytesIO(image_bytes))
                images.append(pil_img)
            
            if images:
                display(Markdown(f"## Diagrams on Page {page_num + 1}\n"))
                for i, img in enumerate(images, 1):
                    display(Markdown(f"### Diagram {i}"))
                    display(img)
                return f"Displayed {len(images)} diagrams from page {page_num + 1}"
            else:
                return f"No diagrams found on page {page_num + 1}"
        except Exception as e:
            return f"Error extracting diagrams: {str(e)}"

# UI for the document extractor
def create_document_writer_ui():
    extractor = PDFDocumentExtractor()
    
    # Create widgets
    upload_btn = widgets.FileUpload(
        description='Upload PDF',
        multiple=False,
        accept='.pdf'
    )
    
    page_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=0,
        step=1,
        description='Page:',
        disabled=True
    )
    
    content_btn = widgets.Button(
        description='Show Page Content',
        disabled=True
    )
    
    diagrams_btn = widgets.Button(
        description='Show Diagrams',
        disabled=True
    )
    
    search_box = widgets.Text(
        placeholder='Enter search term...',
        description='Search:',
        disabled=True
    )
    
    search_btn = widgets.Button(
        description='Search',
        disabled=True
    )
    
    result_selector = widgets.Dropdown(
        options=[],
        description='Result:',
        disabled=True
    )
    
    show_result_btn = widgets.Button(
        description='Show Result',
        disabled=True
    )
    
    output = widgets.Output()
    
    # Update functions
    def update_page_slider():
        if extractor.current_doc:
            page_slider.max = len(extractor.current_doc) - 1
            page_slider.disabled = False
            content_btn.disabled = False
            diagrams_btn.disabled = False
            search_box.disabled = False
            search_btn.disabled = False
    
    def on_upload_change(change):
        if upload_btn.value:
            with output:
                output.clear_output()
                file_name = next(iter(upload_btn.value))
                file_content = upload_btn.value[file_name]['content']
                
                # Save to temporary file
                with open(file_name, 'wb') as f:
                    f.write(file_content)
                
                result = extractor.load_pdf(file_name)
                print(result)
                update_page_slider()
    
    def on_page_change(change):
        extractor.current_page = change['new']
    
    def on_content_click(b):
        with output:
            output.clear_output()
            print(extractor.extract_page_content(page_slider.value))
    
    def on_diagrams_click(b):
        with output:
            output.clear_output()
            print(extractor.get_diagrams(page_slider.value))
    
    def on_search_click(b):
        with output:
            output.clear_output()
            if search_box.value:
                result = extractor.search_content(search_box.value)
                print(result)
                
                if extractor.search_results:
                    result_selector.options = [
                        (f"Result {i+1} (Page {r['page']+1})", i+1) 
                        for i, r in enumerate(extractor.search_results)
                    ]
                    result_selector.disabled = False
                    show_result_btn.disabled = False
    
    def on_show_result_click(b):
        with output:
            output.clear_output()
            print(extractor.show_search_result(result_selector.value))
    
    # Assign handlers
    upload_btn.observe(on_upload_change, names='value')
    page_slider.observe(on_page_change, names='value')
    content_btn.on_click(on_content_click)
    diagrams_btn.on_click(on_diagrams_click)
    search_btn.on_click(on_search_click)
    show_result_btn.on_click(on_show_result_click)
    
    # Layout
    controls = widgets.VBox([
        widgets.HBox([upload_btn]),
        widgets.HBox([page_slider, content_btn, diagrams_btn]),
        widgets.HBox([search_box, search_btn]),
        widgets.HBox([result_selector, show_result_btn]),
        output
    ])
    
    display(controls)

# Run the interface
create_document_writer_ui()
