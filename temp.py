# Advanced PDF Question Answering System for Google Colab
!pip install PyMuPDF sentence-transformers matplotlib scikit-learn Pillow ipywidgets nltk pdf2image
!apt-get install poppler-utils  # Required for pdf2image

import os
import re
import fitz  # PyMuPDF
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from PIL import Image
import io
from google.colab import files
import IPython.display as ip_display
from ipywidgets import widgets, Layout
from IPython.display import display as ipy_display
from nltk.tokenize import sent_tokenize
from pdf2image import convert_from_bytes
import nltk

# =============================================
# NLTK RESOURCE DOWNLOAD - MUST RUN FIRST
# =============================================
print("Downloading NLTK resources...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Explicitly download punkt_tab if needed
try:
    nltk.data.find('tokenizers/punkt/PY3/english.pickle')
except LookupError:
    print("Downloading punkt_tab resources...")
    nltk.download('punkt', quiet=True)

# Verify all resources are available
try:
    sent_tokenize("Test sentence.")
    from nltk.corpus import stopwords
    stopwords.words('english')
    print("All NLTK resources verified successfully!")
except LookupError as e:
    print(f"Error verifying NLTK resources: {e}")
    print("Trying full download...")
    nltk.download('all', quiet=True)
# =============================================

from nltk.corpus import stopwords

class AdvancedPDFProcessor:
    def __init__(self, model_name="all-mpnet-base-v2"):
        """Initialize with verified NLTK resources"""
        # Double-check NLTK resources
        try:
            sent_tokenize("Test sentence.")
            self.stopwords = set(stopwords.words('english'))
        except LookupError as e:
            print("Critical NLTK resources missing, attempting download...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            self.stopwords = set(stopwords.words('english'))
            
        self.model = SentenceTransformer(model_name)
        self.pdf_data = {}
        self.tfidf_vectorizer = TfidfVectorizer(
            max_df=0.85, min_df=2, stop_words='english', ngram_range=(1, 2))
        self.section_embeddings = {}
        self.diagrams = {}
        self.tables = {}
        self.metadata = {}
        self.references = {}
        self.last_results = None
        self.document_structure = {}
        self.stopwords = set(stopwords.words('english'))

    def load_pdfs(self, pdf_directory: str) -> None:
        """
        Load all PDFs from a directory and process them with advanced techniques.
        """
        for filename in os.listdir(pdf_directory):
            if filename.endswith('.pdf'):
                file_path = os.path.join(pdf_directory, filename)
                self._process_pdf(file_path)
    
    def process_uploaded_pdf(self, file_path: str) -> None:
        """Process a single uploaded PDF file with advanced features."""
        self._process_pdf(file_path)
        print(f"Advanced processing complete for: {os.path.basename(file_path)}")
        self._analyze_document_structure(os.path.basename(file_path))
        
    def _process_pdf(self, pdf_path: str) -> None:
        """
        Advanced PDF processing with improved text extraction, structure analysis,
        and metadata handling.
        """
        doc = fitz.open(pdf_path)
        filename = os.path.basename(pdf_path)
        
        # Initialize document data with enhanced structure
        self.pdf_data[filename] = {
            'title': self._extract_document_title(doc),
            'authors': [],
            'abstract': '',
            'sections': [],
            'subsections': defaultdict(list),
            'text': "",
            'pages': len(doc),
            'metadata': {},
            'references': [],
            'tables': [],
            'equations': [],
            'figures': []
        }
        
        # Extract metadata
        self._extract_metadata(doc, filename)
        
        # Extract document structure with hierarchy
        current_section = {'title': '', 'level': 0, 'content': '', 'page': 0}
        full_text = []
        
        # Process each page with advanced features
        for page_num, page in enumerate(doc):
            # Extract text with formatting information
            text_blocks = page.get_text("dict")["blocks"]
            page_text = ""
            
            for block in text_blocks:
                if block['type'] == 0:  # Text block
                    for line in block["lines"]:
                        for span in line["spans"]:
                            # Detect section headers based on font properties
                            if self._is_section_header(span):
                                # Save previous section if it has content
                                if current_section['title'] and current_section['content']:
                                    self._add_section(filename, current_section.copy())
                                
                                # Start new section
                                current_section = {
                                    'title': span['text'].strip(),
                                    'level': self._determine_header_level(span),
                                    'content': '',
                                    'page': page_num + 1,
                                    'font': span['font'],
                                    'size': span['size']
                                }
                            else:
                                current_section['content'] += span['text'] + ' '
                    page_text += ' '.join([span['text'] for line in block["lines"] for span in line["spans"]]) + '\n'
                
                # Handle other block types (images, tables)
                elif block['type'] == 1:  # Image block
                    self._process_image_block(block, filename, page_num)
                
            full_text.append(page_text)
            
            # Add the last section if it exists
            if current_section['title'] and current_section['content']:
                self._add_section(filename, current_section.copy())
        
        # Post-processing
        self.pdf_data[filename]['text'] = '\n'.join(full_text)
        self._extract_references(filename)
        self._create_embeddings(filename)
        self._create_document_summary(filename)
        
    def _is_section_header(self, span: Dict) -> bool:
        """Advanced header detection using font properties and text patterns."""
        text = span['text'].strip()
        
        # Check common header patterns
        header_patterns = [
            r'^[0-9]+(\.[0-9]+)*\s+[A-Z]',  # 1.1 Section
            r'^[A-Z][A-Z\s]{5,}',  # ALL CAPS HEADER
            r'^[IVX]+\.',  # Roman numerals
            r'^Appendix\s+[A-Z0-9]',  # Appendix A
            r'^Abstract$|^Introduction$|^References?$|^Conclusion$|^Acknowledgements?$'
        ]
        
        # Check if text matches any header pattern and has larger font size
        return (any(re.match(pattern, text) for pattern in header_patterns) or 
                span['size'] > 11)  # Assuming body text is <= 11pt
    
    def _determine_header_level(self, span: Dict) -> int:
        """Determine header level based on numbering and font size."""
        text = span['text'].strip()
        
        # Level based on numbering
        if re.match(r'^\d+\.\d+\.\d+', text):
            return 3
        elif re.match(r'^\d+\.\d+', text):
            return 2
        elif re.match(r'^\d+', text):
            return 1
        
        # Fallback to font size
        if span['size'] > 14:
            return 1
        elif span['size'] > 12:
            return 2
        return 3
    
    def _add_section(self, filename: str, section: Dict) -> None:
        """Add section to document structure with hierarchy."""
        self.pdf_data[filename]['sections'].append(section)
        
        # Organize sections by level
        if section['level'] == 1:
            self.pdf_data[filename]['subsections']['main'].append(section)
        else:
            # Find parent section
            last_main = self.pdf_data[filename]['subsections']['main'][-1] if self.pdf_data[filename]['subsections']['main'] else None
            if last_main:
                self.pdf_data[filename]['subsections'][last_main['title']].append(section)
    
    def _process_image_block(self, block: Dict, filename: str, page_num: int) -> None:
        """Advanced image processing with caption detection."""
        try:
            # Extract image
            img = block['image']
            image_bytes = img['image']
            
            # Generate a unique key
            img_key = f"{filename}_page{page_num+1}_img{len(self.diagrams)+1}"
            
            # Try to extract caption (look in surrounding text)
            caption = self._extract_caption_advanced(block, page_num)
            
            # Store with enhanced metadata
            self.diagrams[img_key] = {
                'image': image_bytes,
                'source': filename,
                'page': page_num + 1,
                'caption': caption,
                'type': self._classify_image_type(caption),
                'position': (block['bbox'][0], block['bbox'][1])
            }
            
            # Add to document figures
            self.pdf_data[filename]['figures'].append({
                'key': img_key,
                'caption': caption,
                'page': page_num + 1
            })
        except Exception as e:
            print(f"Error processing image: {e}")
    
    def _extract_caption_advanced(self, block: Dict, page_num: int) -> str:
        """Improved caption extraction using spatial relationships."""
        # This is a placeholder - in a real system you'd implement:
        # 1. Text block proximity analysis
        # 2. Common caption patterns ("Figure 1:", "Table 2:")
        # 3. Font size/style matching
        return "Untitled Figure"
    
    def _classify_image_type(self, caption: str) -> str:
        """Classify image as figure, diagram, table, etc."""
        caption_lower = caption.lower()
        if 'table' in caption_lower:
            return 'table'
        elif 'diagram' in caption_lower or 'chart' in caption_lower:
            return 'diagram'
        elif 'graph' in caption_lower or 'plot' in caption_lower:
            return 'graph'
        return 'figure'
    
    def _extract_metadata(self, doc: Any, filename: str) -> None:
        """Extract document metadata (title, authors, abstract)."""
        first_page = doc[0].get_text()
        lines = [line.strip() for line in first_page.split('\n') if line.strip()]
        
        # Extract title (usually first line)
        if lines:
            self.pdf_data[filename]['title'] = lines[0]
        
        # Extract authors (lines before abstract)
        author_lines = []
        for line in lines[1:]:
            if 'abstract' not in line.lower():
                author_lines.append(line)
            else:
                break
        self.pdf_data[filename]['authors'] = author_lines
        
        # Extract abstract (text after "Abstract" heading)
        abstract_start = None
        for i, line in enumerate(lines):
            if 'abstract' in line.lower():
                abstract_start = i
                break
        if abstract_start:
            self.pdf_data[filename]['abstract'] = ' '.join(lines[abstract_start+1:])
    
    def _extract_document_title(self, doc: Any) -> str:
        """Extract document title from metadata or first page."""
        if 'title' in doc.metadata and doc.metadata['title']:
            return doc.metadata['title']
        
        first_page = doc[0].get_text()
        if first_page:
            first_line = first_page.split('\n')[0].strip()
            if first_line and len(first_line.split()) < 15:  # Reasonable title length
                return first_line
        return os.path.basename(doc.name)
    
    def _extract_references(self, filename: str) -> None:
        """Identify and extract references section."""
        text = self.pdf_data[filename]['text']
        
        # Find references section
        ref_section = None
        for section in self.pdf_data[filename]['sections']:
            if 'reference' in section['title'].lower():
                ref_section = section
                break
        
        if ref_section:
            # Simple reference extraction (can be enhanced with proper parsing)
            references = []
            for line in ref_section['content'].split('\n'):
                if line.strip() and len(line.strip()) > 20:  # Basic filter
                    references.append(line.strip())
            self.pdf_data[filename]['references'] = references
    
    def _create_embeddings(self, filename: str) -> None:
        """Create embeddings for sections with hierarchical context."""
        for i, section in enumerate(self.pdf_data[filename]['sections']):
            # Include parent section titles in embedding
            context = ""
            if section['level'] > 1:
                # Find parent sections
                for parent in self.pdf_data[filename]['sections']:
                    if parent['level'] < section['level']:
                        context += parent['title'] + " "
            
            section_text = context + section['title'] + "\n" + section['content']
            embedding = self.model.encode(section_text)
            
            key = f"{filename}_section{i}"
            self.section_embeddings[key] = {
                'embedding': embedding,
                'title': section['title'],
                'pdf': filename,
                'page': section['page'],
                'level': section['level'],
                'content': section['content']
            }
    
    def _create_document_summary(self, filename: str) -> None:
        """Generate a structured summary of the document."""
        summary = {
            'title': self.pdf_data[filename]['title'],
            'authors': self.pdf_data[filename]['authors'],
            'abstract': self.pdf_data[filename]['abstract'],
            'sections': [s['title'] for s in self.pdf_data[filename]['sections'] if s['level'] == 1],
            'figures': len(self.pdf_data[filename]['figures']),
            'tables': len([img for img in self.diagrams.values() if img['type'] == 'table']),
            'key_topics': self._extract_key_topics(filename)
        }
        self.pdf_data[filename]['summary'] = summary
    
    def _extract_key_topics(self, filename: str) -> List[str]:
        """Extract key topics using TF-IDF."""
        text = self.pdf_data[filename]['text']
        sentences = sent_tokenize(text)
        
        # Use TF-IDF to find important terms
        vectorizer = TfidfVectorizer(max_features=20, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        feature_array = np.array(vectorizer.get_feature_names_out())
        tfidf_sorting = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]
        
        return list(feature_array[tfidf_sorting][:10])  # Top 10 terms
    
    def _analyze_document_structure(self, filename: str) -> None:
        """Analyze and store document structure information."""
        doc_info = {
            'section_hierarchy': self._get_section_hierarchy(filename),
            'figure_distribution': self._get_figure_distribution(filename),
            'section_lengths': self._get_section_lengths(filename),
            'reading_time': self._estimate_reading_time(filename)
        }
        self.document_structure[filename] = doc_info
    
    def _get_section_hierarchy(self, filename: str) -> Dict:
        """Extract section hierarchy with levels."""
        hierarchy = {}
        for section in self.pdf_data[filename]['sections']:
            if section['level'] == 1:
                hierarchy[section['title']] = []
            elif section['level'] == 2 and hierarchy:
                last_main = list(hierarchy.keys())[-1]
                hierarchy[last_main].append(section['title'])
        return hierarchy
    
    def _get_figure_distribution(self, filename: str) -> Dict:
        """Analyze figure distribution across sections."""
        distribution = defaultdict(int)
        for figure in self.pdf_data[filename]['figures']:
            # Find which section the figure is in
            for section in self.pdf_data[filename]['sections']:
                if figure['page'] >= section['page']:
                    distribution[section['title']] += 1
        return dict(distribution)
    
    def _get_section_lengths(self, filename: str) -> Dict:
        """Calculate word counts for each section."""
        lengths = {}
        for section in self.pdf_data[filename]['sections']:
            lengths[section['title']] = len(section['content'].split())
        return lengths
    
    def _estimate_reading_time(self, filename: str) -> int:
        """Estimate reading time in minutes (average 200 wpm)."""
        word_count = len(self.pdf_data[filename]['text'].split())
        return max(1, round(word_count / 200))
    
    def search(self, query: str, top_k: int = 5, include_content: bool = False) -> List[Dict]:
        """Advanced search with content inclusion option."""
        query_embedding = self.model.encode(query)
        results = []
        
        for key, data in self.section_embeddings.items():
            similarity = np.dot(query_embedding, data['embedding']) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(data['embedding']))
            
            result = {
                'key': key,
                'similarity': similarity,
                'title': data['title'],
                'pdf': data['pdf'],
                'page': data['page'],
                'level': data['level']
            }
            
            if include_content:
                result['content'] = data['content'][:500] + "..."  # First 500 chars
            
            results.append(result)
        
        # Store last results for follow-up questions
        self.last_results = sorted(results, key=lambda x: x['similarity'], reverse=True)[:top_k]
        return self.last_results
    
    def find_diagram(self, query: str, top_k: int = 3) -> List[Dict]:
        """Enhanced diagram search with type filtering."""
        # Check if query specifies diagram type
        diagram_type = None
        type_keywords = {
            'table': ['table', 'tabular'],
            'diagram': ['diagram', 'flowchart', 'schematic'],
            'graph': ['graph', 'plot', 'chart'],
            'figure': ['figure', 'image', 'illustration']
        }
        
        for t, keywords in type_keywords.items():
            if any(kw in query.lower() for kw in keywords):
                diagram_type = t
                break
        
        # Filter by type if specified
        if diagram_type:
            filtered_diagrams = {k: v for k, v in self.diagrams.items() 
                               if v['type'] == diagram_type}
        else:
            filtered_diagrams = self.diagrams
        
        # Proceed with search on filtered diagrams
        captions = [data['caption'] for data in filtered_diagrams.values()]
        if not captions:
            return []
        
        caption_embeddings = self.model.encode(captions)
        query_embedding = self.model.encode(query)
        
        similarities = np.dot(caption_embeddings, query_embedding) / (
            np.linalg.norm(caption_embeddings, axis=1) * np.linalg.norm(query_embedding))
        
        results = []
        for idx, key in enumerate(filtered_diagrams.keys()):
            if idx < len(similarities):
                diagram_data = filtered_diagrams[key]
                results.append({
                    'key': key,
                    'similarity': similarities[idx],
                    'caption': diagram_data['caption'],
                    'pdf': diagram_data['source'],
                    'page': diagram_data['page'],
                    'type': diagram_data['type']
                })
        
        self.last_diagram_results = sorted(results, key=lambda x: x['similarity'], reverse=True)[:top_k]
        return self.last_diagram_results
    
    def generate_diagram(self, diagram_key: str, size: Tuple[int, int] = (800, 600)) -> Optional[Image.Image]:
        """Generate diagram with size control."""
        if diagram_key in self.diagrams:
            image_bytes = self.diagrams[diagram_key]['image']
            img = Image.open(io.BytesIO(image_bytes))
            img.thumbnail(size, Image.Resampling.LANCZOS)
            return img
        return None
    
    def get_document_summary(self, pdf_name: str) -> Dict:
        """Get structured summary of a document."""
        if pdf_name in self.pdf_data:
            return self.pdf_data[pdf_name]['summary']
        return {}
    
    def visualize_document_structure(self, pdf_name: str):
        """Create visualization of document structure."""
        if pdf_name not in self.document_structure:
            return None
        
        doc_info = self.document_structure[pdf_name]
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Document Structure Analysis: {pdf_name}", fontsize=16)
        
        # Section hierarchy visualization
        self._plot_section_hierarchy(axes[0, 0], doc_info['section_hierarchy'])
        
        # Figure distribution
        self._plot_figure_distribution(axes[0, 1], doc_info['figure_distribution'])
        
        # Section lengths
        self._plot_section_lengths(axes[1, 0], doc_info['section_lengths'])
        
        # Reading metrics
        self._plot_reading_metrics(axes[1, 1], doc_info)
        
        plt.tight_layout()
        return fig
    
    def _plot_section_hierarchy(self, ax, hierarchy):
        """Plot section hierarchy as a tree."""
        ax.set_title("Section Hierarchy")
        
        y_pos = 0
        for main_section, subsections in hierarchy.items():
            ax.text(0.1, y_pos, main_section, fontsize=12, fontweight='bold')
            y_pos -= 1
            for sub in subsections:
                ax.text(0.2, y_pos, sub, fontsize=10)
                y_pos -= 0.8
        
        ax.set_xlim(0, 1)
        ax.set_ylim(y_pos, 1)
        ax.axis('off')
    
    def _plot_figure_distribution(self, ax, distribution):
        """Plot figure distribution across sections."""
        if not distribution:
            ax.text(0.5, 0.5, "No figures found", ha='center')
            ax.axis('off')
            return
        
        sections = list(distribution.keys())
        counts = list(distribution.values())
        
        ax.barh(sections, counts, color='skyblue')
        ax.set_title("Figure Distribution by Section")
        ax.set_xlabel("Number of Figures")
        ax.set_ylabel("Sections")
        plt.setp(ax.get_yticklabels(), fontsize=8)
    
    def _plot_section_lengths(self, ax, lengths):
        """Plot section lengths (word counts)."""
        if not lengths:
            ax.text(0.5, 0.5, "No sections found", ha='center')
            ax.axis('off')
            return
        
        sections = list(lengths.keys())
        words = list(lengths.values())
        
        ax.barh(sections, words, color='lightgreen')
        ax.set_title("Section Lengths (Word Count)")
        ax.set_xlabel("Word Count")
        ax.set_ylabel("Sections")
        plt.setp(ax.get_yticklabels(), fontsize=8)
    
    def _plot_reading_metrics(self, ax, doc_info):
        """Display reading metrics."""
        text = f"""
        Document Metrics:
        - Estimated Reading Time: {doc_info['reading_time']} minutes
        - Main Sections: {len(doc_info['section_hierarchy'])}
        - Subsections: {sum(len(subs) for subs in doc_info['section_hierarchy'].values())}
        - Figures: {sum(doc_info['figure_distribution'].values())}
        """
        
        ax.text(0.1, 0.5, text, fontsize=12)
        ax.set_title("Reading Metrics")
        ax.axis('off')
    
    def answer_query(self, query: str) -> str:
        """Advanced query answering with context awareness."""
        # Check for specific commands
        if query.lower().startswith('summary of '):
            pdf_name = query[11:].strip()
            return self._generate_document_summary_response(pdf_name)
        
        if "structure of " in query.lower():
            pdf_name = query.lower().split('structure of ')[1].strip()
            return self._generate_structure_response(pdf_name)
        
        # Check for diagram requests
        if any(keyword in query.lower() for keyword in ['diagram', 'figure', 'image', 'table', 'graph']):
            return self._handle_diagram_query(query)
        
        # Regular content search
        return self._handle_content_query(query)
    
    def _generate_document_summary_response(self, pdf_name: str) -> str:
        """Generate a detailed summary response for a document."""
        if pdf_name not in self.pdf_data:
            return f"I couldn't find a PDF named '{pdf_name}' in the loaded documents."
        
        summary = self.get_document_summary(pdf_name)
        response = f"# Document Summary: {summary['title']}\n\n"
        response += f"**Authors:** {', '.join(summary['authors'])}\n\n"
        response += f"**Abstract:** {summary['abstract'][:300]}...\n\n"
        response += "## Main Sections:\n"
        for i, section in enumerate(summary['sections'], 1):
            response += f"{i}. {section}\n"
        
        response += f"\n**Key Topics:** {', '.join(summary['key_topics'])}\n"
        response += f"**Figures/Tables:** {summary['figures']} figures, {summary['tables']} tables\n"
        response += f"**Estimated Reading Time:** {summary['reading_time']} minutes\n"
        
        return response
    
    def _generate_structure_response(self, pdf_name: str) -> str:
        """Generate a response about document structure."""
        if pdf_name not in self.document_structure:
            return f"I couldn't analyze the structure of '{pdf_name}'."
        
        doc_info = self.document_structure[pdf_name]
        response = f"# Document Structure: {pdf_name}\n\n"
        
        # Section hierarchy
        response += "## Section Hierarchy:\n"
        for main_section, subsections in doc_info['section_hierarchy'].items():
            response += f"- {main_section}\n"
            for sub in subsections:
                response += f"  - {sub}\n"
        
        # Figure distribution
        response += "\n## Figure Distribution:\n"
        for section, count in doc_info['figure_distribution'].items():
            response += f"- {section}: {count} figures\n"
        
        # Reading time
        response += f"\n**Estimated Reading Time:** {doc_info['reading_time']} minutes\n"
        
        # Offer visualization
        response += "\nYou can view a detailed visualization by requesting: 'Show structure visualization'"
        
        return response
    
    def _handle_diagram_query(self, query: str) -> str:
        """Handle queries about diagrams/figures/tables."""
        results = self.find_diagram(query)
        
        if not results:
            return "I couldn't find any relevant diagrams matching your query."
        
        response = "I found these relevant visual elements:\n\n"
        for i, res in enumerate(results, 1):
            response += f"{i}. [{res['type'].upper()}] {res['caption']} (from {res['pdf']}, page {res['page']})\n"
        
        response += "\nYou can view a specific item by requesting: 'Show item 1'"
        return response
    
    def _handle_content_query(self, query: str) -> str:
        """Handle regular content queries."""
        results = self.search(query, include_content=True)
        
        if not results:
            return "I couldn't find any relevant information about that in the PDFs."
        
        response = "Here are the most relevant sections I found:\n\n"
        for i, result in enumerate(results, 1):
            response += f"{i}. **{result['title']}** (Level {result['level']})\n"
            response += f"   Source: {result['pdf']}, Page {result['page']}\n"
            response += f"   Content: {result.get('content', '')}\n\n"
        
        response += "You can request more details about a specific section by asking: 'Tell me more about section 1'"
        return response
    
    def get_section_details(self, section_index: int) -> str:
        """Get full details of a specific section from last results."""
        if not self.last_results or section_index > len(self.last_results):
            return "Invalid section index or no query was made recently."
        
        result = self.last_results[section_index - 1]
        content = result.get('content', '')
        
        # If we don't have full content, try to get it
        if len(content) < 100:
            for key, data in self.section_embeddings.items():
                if data['pdf'] == result['pdf'] and data['title'] == result['title']:
                    content = data['content']
                    break
        
        return f"### {result['title']}\n\n**Source:** {result['pdf']}, Page {result['page']}\n\n{content}"
    
    def show_diagram(self, diagram_index: int) -> Tuple[Optional[Image.Image], str]:
        """Show a specific diagram with enhanced display options."""
        if not self.last_diagram_results or diagram_index > len(self.last_diagram_results):
            return None, "Invalid diagram index or no diagram query was made recently."
        
        result = self.last_diagram_results[diagram_index - 1]
        diagram_key = result['key']
        
        img = self.generate_diagram(diagram_key, size=(1000, 800))
        if img:
            reference = f"{result['type'].upper()}: {result['caption']}\nFrom {result['pdf']}, page {result['page']}"
            return img, reference
        else:
            return None, "Failed to retrieve the diagram."


class AdvancedPDFQASystem:
    def __init__(self):
        """Initialize the advanced PDF QA system."""
        self.processor = AdvancedPDFProcessor()
        self.chat_history = []
        
    def upload_pdfs(self):
        """Enhanced PDF upload with progress tracking."""
        print("Please select PDF files to upload...")
        uploaded = files.upload()
        
        if not uploaded:
            print("No files were uploaded.")
            return
        
        print("\nProcessing uploaded files...")
        progress = widgets.IntProgress(
            value=0,
            min=0,
            max=len(uploaded),
            description='Processing:',
            bar_style='info',
            style={'bar_color': '#4285F4'},
            orientation='horizontal'
        )
        ip_display.display(progress)  # Changed from display() to ip_display.display()
        
        for i, (filename, content) in enumerate(uploaded.items()):
            if filename.endswith('.pdf'):
                # Save and process
                with open(filename, 'wb') as f:
                    f.write(content)
                self.processor.process_uploaded_pdf(filename)
                progress.value = i + 1
        
        print(f"\nProcessing complete. Loaded {len(self.processor.pdf_data)} PDF(s).")
        self._show_loaded_documents()
    
    def _show_loaded_documents(self):
        """Display loaded documents in an organized table."""
        from IPython.display import HTML
        
        if not self.processor.pdf_data:
            print("No documents loaded.")
            return
        
        table_html = """
        <style>
            .doc-table {
                width: 100%;
                border-collapse: collapse;
                margin: 10px 0;
                font-family: Arial, sans-serif;
            }
            .doc-table th {
                background-color: #4285F4;
                color: white;
                text-align: left;
                padding: 8px;
            }
            .doc-table td {
                border: 1px solid #ddd;
                padding: 8px;
            }
            .doc-table tr:nth-child(even) {
                background-color: #f2f2f2;
            }
        </style>
        <table class="doc-table">
            <tr>
                <th>Document</th>
                <th>Title</th>
                <th>Pages</th>
                <th>Sections</th>
                <th>Figures</th>
            </tr>
        """
        
        for pdf_name, data in self.processor.pdf_data.items():
            table_html += f"""
            <tr>
                <td>{pdf_name}</td>
                <td>{data['title']}</td>
                <td>{data['pages']}</td>
                <td>{len(data['sections'])}</td>
                <td>{len(data['figures'])}</td>
            </tr>
            """
        
        table_html += "</table>"
        ip_display.display(HTML(table_html))
    
    def process_query(self, query: str) -> Any:
        """
        Process a user query with advanced features and return a response.
        Returns either text response or displays visualizations.
        """
        # Add to chat history
        self.chat_history.append(('user', query))
        
        # Check for special commands
        if query.lower() == 'list documents':
            return self._show_loaded_documents()
        
        if query.lower().startswith('show structure visualization'):
            pdf_name = query[27:].strip() or next(iter(self.processor.pdf_data.keys()), None)
            if pdf_name in self.processor.pdf_data:
                fig = self.processor.visualize_document_structure(pdf_name)
                if fig:
                    plt.close()
                    return fig
                return "No structure visualization available for this document."
            return "Please specify a valid document name."
        
        # Check for diagram display requests
        show_diagram_match = re.match(r"show\s+(item|diagram|figure|table)\s+(\d+)", query.lower())
        if show_diagram_match:
            diagram_index = int(show_diagram_match.group(2))
            img, reference = self.processor.show_diagram(diagram_index)
            if img:
                # Display with enhanced formatting
                plt.figure(figsize=(12, 8))
                plt.imshow(img)
                plt.axis('off')
                plt.title(reference, pad=20)
                plt.tight_layout()
                plt.show()
                
                # Add to chat history
                self.chat_history.append(('system', f"Displayed diagram {diagram_index}"))
                return f"Displayed item {diagram_index}. {reference}"
            else:
                return reference
        
        # Check for section detail requests
        section_details_match = re.match(r"tell\s+me\s+more\s+about\s+(section|item)\s+(\d+)", query.lower())
        if section_details_match:
            section_index = int(section_details_match.group(2))
            details = self.processor.get_section_details(section_index)
            
            # Add to chat history
            self.chat_history.append(('system', f"Provided details for section {section_index}"))
            return details
        
        # Process general queries
        response = self.processor.answer_query(query)
        
        # Add to chat history
        self.chat_history.append(('system', response))
        return response
    
    def show_chat_history(self):
        """Display the conversation history in a formatted way."""
        from IPython.display import HTML
        
        if not self.chat_history:
            return "No conversation history yet."
        
        history_html = """
        <style>
            .chat-container {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
            }
            .chat-message {
                margin: 10px 0;
                padding: 12px;
                border-radius: 8px;
                line-height: 1.4;
            }
            .user-message {
                background-color: #e3f2fd;
                border-left: 4px solid #4285F4;
            }
            .system-message {
                background-color: #f1f1f1;
                border-left: 4px solid #34A853;
            }
            .message-label {
                font-weight: bold;
                margin-bottom: 5px;
                color: #555;
            }
        </style>
        <div class="chat-container">
        """
        
        for i, (sender, message) in enumerate(self.chat_history, 1):
            if sender == 'user':
                history_html += f"""
                <div class="chat-message user-message">
                    <div class="message-label">You (message {i}):</div>
                    <div>{message}</div>
                </div>
                """
            else:
                # Convert markdown-style headers to HTML
                message = message.replace('### ', '<h3>').replace('\n\n', '</h3>\n\n')
                message = message.replace('## ', '<h2>').replace('\n\n', '</h2>\n\n')
                message = message.replace('# ', '<h1>').replace('\n\n', '</h1>\n\n')
                
                history_html += f"""
                <div class="chat-message system-message">
                    <div class="message-label">Assistant (message {i}):</div>
                    <div>{message}</div>
                </div>
                """
        
        history_html += "</div>"
        ipy_display(HTML(history_html))

def run_advanced_pdf_qa_system():
    """Run with guaranteed NLTK resources"""
    # Final verification
    try:
        sent_tokenize("Test sentence.")
        stopwords.words('english')
    except LookupError:
        print("Critical error: NLTK resources still missing after all attempts")
        return
    
    qa_system = AdvancedPDFQASystem()
    
    # Create UI elements
    header = widgets.HTML(
        "<h1 style='color: #4285F4; font-family: Arial, sans-serif;'>Advanced PDF Question Answering System</h1>"
        "<p style='font-size: 16px;'>Upload PDFs and ask questions about their content.</p>"
    )
    
    upload_btn = widgets.Button(
        description="Upload PDFs",
        button_style='primary',
        icon='upload',
        layout=Layout(width='150px', height='40px')
    )
    
    history_btn = widgets.Button(
        description="Show History",
        button_style='info',
        icon='history',
        layout=Layout(width='150px', height='40px')
    )
    
    query_box = widgets.Textarea(
        value='',
        placeholder='Ask me about the PDFs, "show item X", or "tell me more about section Y"...',
        description='Query:',
        disabled=False,
        layout=Layout(width='80%', height='80px'),
        style={'description_width': 'initial'}
    )
    
    output = widgets.Output(layout={'border': '1px solid #ddd', 'padding': '10px'})
    
    # Define button actions
    def on_upload_click(b):
        with output:
            output.clear_output()
            qa_system.upload_pdfs()
    
    def on_history_click(b):
        with output:
            output.clear_output()
            qa_system.show_chat_history()
    
    def on_submit(b):
        query = query_box.value
        with output:
            output.clear_output()
            if query.strip():
                print(f"\nYour query: {query}")
                response = qa_system.process_query(query)
                
                if isinstance(response, plt.Figure):
                    plt.close()
                    ipy_display(response)
                else:
                    print("\nResponse:")
                    ipy_display(response)
                
                print("\n" + "-"*50)
        query_box.value = ''
    
    # Assign click handlers
    upload_btn.on_click(on_upload_click)
    history_btn.on_click(on_history_click)
    
    submit_btn = widgets.Button(
        description="Submit",
        button_style='success',
        icon='search',
        layout=Layout(width='150px', height='40px')
    )
    submit_btn.on_click(on_submit)
    
    # Create UI layout - Using widgets.VBox directly
    controls = widgets.VBox([upload_btn, history_btn])
    query_area = widgets.VBox([query_box, submit_btn])
    
    # Display the full interface using the renamed display function
    ipy_display(widgets.VBox([header, controls, query_area, output]))

# Run the advanced system
run_advanced_pdf_qa_system()
