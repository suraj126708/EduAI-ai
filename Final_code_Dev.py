from fastapi import FastAPI, HTTPException, UploadFile, Form, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from typing import List, Dict, Optional
from copy import deepcopy

import os
import json
import re
import time
import concurrent.futures
import io
import logging
import traceback
from pathlib import Path
import tempfile

from dotenv import load_dotenv
load_dotenv()

# ---- Imports for Hugging Face Deployment ----
import base64
import google.generativeai as genai
import google.generativeai as genai
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import HfHubHTTPError
# ---------------------------------------------

# ---- Configure Logging ----
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # <-- Only log to the console
    ]
)
logger = logging.getLogger(__name__)

from PIL import Image
from pdf2image import convert_from_path

# ---- LangChain / Embeddings / Vector Store ----
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# Google Vertex AI
import vertexai
from vertexai.generative_models import GenerativeModel, Part

import platform

try:
    # Note: Initialization will be finalized in initialize_app()
    # We just try to init here to catch immediate errors
    # The GOOGLE_APPLICATION_CREDENTIALS will be set dynamically
    vertexai.init(project="gen-lang-client-0238295665", location="us-central1")
    logger.info("Vertex AI initialized (pre-check)")
except Exception as e:
    logger.warning(f"Failed to pre-init Vertex AI (will retry in initialize_app): {e}")
    # Do not raise here, let initialize_app handle it

def pil_to_part(image, max_size=(1600, 1600), quality=70):
    """Convert PIL Image to Vertex AI Part with optimized compression and resizing"""
    try:
        if image is None:
            raise ValueError("Image cannot be None")
        
        # Resize if image is too large (reduces upload time and processing)
        # Reduced from 2048 to 1600 for faster processing (still sufficient for text extraction)
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        buffer = io.BytesIO()
        # Reduced quality from 75 to 70 for faster uploads (still good quality for text)
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        return Part.from_data(buffer.getvalue(), mime_type="image/jpeg")
    except Exception as e:
        logger.error(f"Error converting PIL image to Vertex AI Part: {e}", exc_info=True)
        raise

# ---- Config ----
# Use a standard path. On HF Spaces, this will be ephemeral.
# We will download/upload to this path from the HF Dataset.
INDEX_DIR = "/tmp/faiss_index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ---- Initialize Embeddings ----
try:
    logger.info(f"Initializing embeddings with model: {MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    logger.info("Embeddings initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize embeddings: {e}", exc_info=True)
    raise

def safe_vertex_generate(model, parts, **kwargs):
    """Optimized Vertex AI generation with faster retry logic and connection optimization"""
    retries = 5
    for attempt in range(retries):
        try:
            if model is None:
                raise ValueError("Model cannot be None")
            if not parts:
                raise ValueError("Parts cannot be empty")
            # Add generation config for faster responses if not provided
            if 'generation_config' not in kwargs:
                from vertexai.generative_models import GenerationConfig
                kwargs['generation_config'] = GenerationConfig(max_output_tokens=8192)
            return model.generate_content(parts, **kwargs)
        except Exception as e:
            error_str = str(e).lower()
            # Faster retry with exponential backoff
            wait_time = min(2 * (2 ** attempt), 30)

            # If Rate Limit (429) or Quota error, wait significantly longer
            if "429" in error_str or "quota" in error_str or "rate limit" in error_str:
                logger.warning(f"Quota exceeded (Attempt {attempt+1}/{retries}). Waiting 30s...")
                time.sleep(30) # Wait 30 seconds for quota to reset
            else:
                logger.warning(f"Vertex AI error (Attempt {attempt+1}/{retries}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                
    logger.error(f"Vertex AI generation failed after {retries} attempts.")
    return None

def extract_multimodal_elements_from_pdf(file_path: str, max_workers: int = 60):
    """
    Robust Extraction with Batch Recovery.
    Fixes: 'single_page_prompt' error and Batch Failures.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")
    
    logger.info(f"Extracting content from: {file_path}")
    
    # 1. Detect Total Pages
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)
    except:
        logger.info("PyPDF2 failed, using fallback for page count.")
        total_pages = None

    try:
        model = GenerativeModel("gemini-2.5-flash")
    except Exception as e:
        logger.error(f"Vertex AI Init failed: {e}")
        raise

    # 2. DEFINING PROMPTS (CRITICAL FIX)
    
    multi_page_prompt = """
    Analyze these {num_pages} textbook pages. Extract content for study material generation.

    RULES:
    1. TEXT: Summarize paragraphs clearly. Keep definitions and key dates exact.
    2. DIAGRAMS: Describe exactly what the diagram shows (e.g., "A diagram showing the human heart with labeled aorta and ventricles").
    3. FORMULAS: detailed mathematical formulas in LaTeX format strictly (e.g. $$ a^2 + b^2 = c^2 $$).
    4. TABLES: Represent as Markdown tables.

    OUTPUT XML FORMAT:
    <pages>
      <page number="1">
        <element type="text">...</element>
        <element type="diagram_caption">...</element>
        <element type="formula_latex">...</element>
      </page>
    </pages>
    
    Return ONLY XML. Use CDATA for special chars if needed.
    """

    # --- ADDED THIS MISSING PROMPT ---
    single_page_prompt = """
    Analyze this single textbook page. Extract content for study material.
    
    RULES:
    1. Summarize text paragraphs.
    2. Describe diagrams/images clearly (tag them as diagram_caption).
    3. Write formulas in LaTeX ($$ ... $$).
    
    OUTPUT XML FORMAT:
    <elements>
      <element type="text">...</element>
      <element type="diagram_caption">...</element>
      <element type="formula_latex">...</element>
    </elements>
    """

    all_elements = []
    batch_size = 150
    pages_per_api_call = 15 

    # --- Robust Processor ---
    def process_group(page_nums, pages):
        if not pages: return []
        
        # Try Batch First
        try:
            parts = [multi_page_prompt.format(num_pages=len(pages))]
            for p in pages:
                parts.append(pil_to_part(p, quality=70))
            
            resp = safe_vertex_generate(model, parts)
            
            if not resp or not resp.candidates: 
                raise ValueError("Empty response")
            
            raw = resp.candidates[0].content.parts[0].text
            data = parse_xml_to_json(raw)
            
            batch_els = []
            if "pages" in data:
                for p_data in data["pages"]:
                    p_num = p_data.get("page_number")
                    if p_num is None and len(page_nums) == len(data["pages"]):
                        p_num = page_nums[data["pages"].index(p_data)]
                    for el in p_data.get("elements", []):
                        el["page"] = p_num
                        batch_els.append(el)
            elif "elements" in data:
                for el in data["elements"]:
                    el["page"] = page_nums[0]
                    batch_els.append(el)
            
            if batch_els: return batch_els
            
        except Exception as e:
            logger.warning(f"Batch {page_nums[0]}-{page_nums[-1]} failed. Retrying one-by-one...")

        # Fallback: One-by-One
        fallback_elements = []
        for p_num, page in zip(page_nums, pages):
            try:
                # Now 'single_page_prompt' is defined and available!
                single_resp = safe_vertex_generate(model, [single_page_prompt, pil_to_part(page)])
                if single_resp and single_resp.candidates:
                    txt = single_resp.candidates[0].content.parts[0].text
                    s_data = parse_xml_to_json(txt)
                    if "elements" in s_data:
                        for el in s_data["elements"]:
                            el["page"] = p_num
                            fallback_elements.append(el)
            except Exception as inner_e:
                logger.error(f"Page {p_num} failed completely: {inner_e}")
                
        return fallback_elements

    # --- Main Loop ---
    current_page = 1
    while True:
        if total_pages and current_page > total_pages: break
        
        end_page = min(current_page + batch_size - 1, total_pages) if total_pages else current_page + batch_size - 1
        logger.info(f"Converting PDF pages {current_page} to {end_page}...")
        
        try:
            pages = convert_from_path(file_path, first_page=current_page, last_page=end_page, dpi=90)
        except Exception as e:
            if not total_pages: break 
            logger.error(f"Conversion error: {e}")
            break
            
        if not pages: break

        groups = []
        for i in range(0, len(pages), pages_per_api_call):
            sub_pages = pages[i : i + pages_per_api_call]
            sub_nums = list(range(current_page + i, current_page + i + len(sub_pages)))
            groups.append((sub_nums, sub_pages))

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_group, nums, pgs): nums for nums, pgs in groups}
            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
                if res: all_elements.extend(res)

        current_page += len(pages)
        if total_pages and current_page > total_pages: break
        if not total_pages and len(pages) < batch_size: break

    logger.info(f"Extraction complete. Found {len(all_elements)} elements.")
    return all_elements

def extract_chapters(file_path: str, num_pages: int = 15, max_workers: int = 10):
    """
    Extracts Chapters. Handles TOCs without page numbers by defaulting to 0.
    Sanitizes output to ensure 'chapter_no' always exists.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")
        
    logger.info(f"Extracting Table of Contents from first {num_pages} pages...")
    
    # Initialize Vertex AI (with fallback)
    try:
        model = GenerativeModel("gemini-2.5-flash")
    except:
        vertexai.init(project="gen-lang-client-0238295665", location="us-central1")
        model = GenerativeModel("gemini-2.5-flash")

    # UPDATED PROMPT: Explicitly handle missing page numbers
    prompt = """
    Analyze this page. Is it a Table of Contents (Index)?

    Avoid confusing or extracting the following pages:
    - Title or cover page
    - Preface, Foreword, Acknowledgement pages
    - Pages with only a single image or poem
    - Pages with no list-like structure or no clear numbering
    
    If YES, extract the chapters in this JSON format:
    [
      {"chapter_no": "1", "chapter_title": "Resources", "start_page": 1},
      {"chapter_no": "2", "chapter_title": "Land and Soil", "start_page": 0} 
    ]
    
    RULES:
    1. Extract the Title and the Page Number listed against it.
    2. **CRITICAL:** If NO page number is visible, set "start_page": 0. Do NOT guess.
    3. Return [] if this is not a TOC page.
    """

    try:
        pages = convert_from_path(file_path, first_page=1, last_page=num_pages, dpi=90)
    except Exception as e:
        logger.error(f"PDF Convert Error: {e}")
        return []

    results = []
    
    def process_toc(page):
        try:
            resp = safe_vertex_generate(model, [prompt, pil_to_part(page)])
            if not resp or not resp.candidates: return []
            
            text = resp.candidates[0].content.parts[0].text
            json_match = re.search(r"\[[\s\S]*\]", text)
            if json_match:
                # Use clean_json_string helper if available, else raw
                raw_json = json_match.group(0)
                try:
                    data = json.loads(clean_json_string(raw_json))
                except:
                    data = json.loads(raw_json)
                return data if isinstance(data, list) else []
        except:
            return []
        return []

    # Scan pages
    for page in pages:
        res = process_toc(page)
        if res:
            results.extend(res)
            # Stop if we found a good TOC to save time
            if len(results) > 2: break 

    # Deduplicate based on title and number
    seen = set()
    unique_chapters = []
    for item in results:
        # Create a unique key (fallback to empty string if keys missing)
        c_no = str(item.get("chapter_no", "")).strip()
        c_title = str(item.get("chapter_title", "")).strip().lower()
        
        key = c_no + c_title
        if key and key not in seen:
            seen.add(key)
            unique_chapters.append(item)
    
    # --- 🟢 NEW SANITIZATION STEP (Fixes the crash) ---
    final_chapters = []
    for i, chap in enumerate(unique_chapters, 1):
        # 1. Ensure chapter_no exists
        if "chapter_no" not in chap or not str(chap["chapter_no"]).strip():
            # Try to extract number from title (e.g. "Chapter 5: Algebra")
            title_match = re.search(r'\b(\d+)\b', str(chap.get("chapter_title", "")))
            if title_match:
                chap["chapter_no"] = title_match.group(1)
            else:
                # Fallback: Use the index (1, 2, 3...)
                chap["chapter_no"] = str(i)
        
        # 2. Ensure chapter_title exists
        if "chapter_title" not in chap or not chap["chapter_title"]:
            chap["chapter_title"] = f"Chapter {chap['chapter_no']}"

        # 3. Ensure start_page is an integer
        try:
            chap["start_page"] = int(chap.get("start_page", 0))
        except:
            chap["start_page"] = 0

        final_chapters.append(chap)
    # --------------------------------------------------

    # Sort by chapter number (handling non-numeric chapters by putting them at end)
    def sorter(x):
        try: 
            return int(re.search(r'\d+', str(x.get("chapter_no", "0"))).group())
        except: 
            return 9999
            
    final_chapters.sort(key=sorter)
    
    logger.info(f"Extracted {len(final_chapters)} chapters.")
    return final_chapters

def create_smart_chunks(subject_data, elements, chapters):
    """
    Refined Chunking with DEBUG LOGS:
    1. Maps pages to chapters.
    2. Groups text BY PAGE first.
    3. Splits large pages into chunks with correct metadata.
    """
    if not elements: return []
    if not chapters: chapters = [{"chapter_no": "1", "chapter_title": "Start", "start_page": 1}]

    logger.info("Creating chunks with Page-Level Granularity...")

    # --- PHASE 1: TITLE HUNTING ---
    for chap in chapters:
        if int(chap.get("start_page", 0)) == 0:
            target_title = re.sub(r'[^a-z0-9]', '', chap.get("chapter_title", "").lower())[:20]
            if len(target_title) < 4: continue
            
            for el in elements:
                content = re.sub(r'[^a-z0-9]', '', el.get("content", "").lower())[:100]
                if target_title in content:
                    chap["start_page"] = el.get("page", 1)
                    # LOGGING ADDED HERE
                    logger.info(f"Found Title '{chap['chapter_title']}' on Page {chap['start_page']}")
                    break

    # --- PHASE 2: MAP PAGES TO CHAPTERS ---
    chapters = sorted(chapters, key=lambda x: int(x.get("start_page", 0)))
    
    last_known_page = 1
    for chap in chapters:
        if int(chap.get("start_page", 0)) == 0: 
            chap["start_page"] = last_known_page + 1
        last_known_page = int(chap["start_page"])

    page_to_chap_map = {}
    for i, chap in enumerate(chapters):
        start = int(chap["start_page"])
        if i < len(chapters) - 1:
            end = int(chapters[i+1]["start_page"]) - 1
        else:
            end = 10000 
        
        # LOGGING ADDED HERE
        logger.info(f"Mapping Chapter {chap['chapter_no']} ({chap['chapter_title']}) to Pages {start}-{end}")
        
        for p in range(start, end + 1):
            page_to_chap_map[p] = chap

    # --- PHASE 3: GROUP CONTENT BY PAGE ---
    page_buffers = {} 

    for el in elements:
        pg = el.get("page", 1)
        content = el.get("content", "")
        
        fig_match = re.search(r'(?:Figure|Fig|Table|Ex)\.?\s*(\d+)', content, re.IGNORECASE)
        forced_chap = None
        if fig_match:
            detected_num = fig_match.group(1)
            forced_chap = next((c for c in chapters if str(c.get("chapter_no")) == detected_num), None)

        if forced_chap:
            assigned_chap = forced_chap
            # LOGGING ADDED HERE (Optional: can be noisy)
            # logger.debug(f"Page {pg}: Figure Intelligence forced Chapter {forced_chap['chapter_no']}")
        else:
            assigned_chap = page_to_chap_map.get(pg, chapters[0] if chapters else {})

        el_type = el.get("type", "text")
        if el_type == "formula_latex": prefix = f"\n[FORMULA]: $$ {content} $$\n"
        elif el_type == "diagram_caption": prefix = f"\n[DIAGRAM]: {content}\n"
        else: prefix = f"{content}\n"

        if pg not in page_buffers:
            page_buffers[pg] = {
                "text": "", 
                "chapter_no": assigned_chap.get("chapter_no", "Unknown"),
                "chapter_title": assigned_chap.get("chapter_title", "Unknown")
            }
        page_buffers[pg]["text"] += prefix

    # --- PHASE 4: CREATE CHUNKS ---
    chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    sorted_pages = sorted(page_buffers.keys())

    for pg in sorted_pages:
        data = page_buffers[pg]
        raw_text = data["text"]
        
        page_splits = text_splitter.split_text(raw_text)
        
        for i, split_text in enumerate(page_splits):
            chunks.append(Document(
                page_content=split_text,
                metadata={
                    **subject_data,
                    "chapter_no": str(data["chapter_no"]),
                    "chapter_name": data["chapter_title"],
                    "page": int(pg),
                    "content_type": "textbook_content",
                    "pdf_name": subject_data.get("pdf_name", "doc"),
                    "chunk_index": i
                }
            ))

    logger.info(f"Created {len(chunks)} chunks from {len(sorted_pages)} pages.")
    return chunks

def process_pdf(subject_data: dict, file_path: str, max_toc_pages: int = 15):
    """
    Process PDF - extracts chapters and all content from the book.
    
    Args:
        subject_data: Dictionary with subject metadata
        file_path: Path to PDF file
        max_toc_pages: Maximum TOC pages to check (default 10)
    
    Returns:
        Tuple of (chapters, chunks)
    
    Raises:
        ValueError: If inputs are invalid
        FileNotFoundError: If PDF file doesn't exist
    """
    if not subject_data or not isinstance(subject_data, dict):
        raise ValueError("subject_data must be a non-empty dictionary")
    
    if not file_path or not isinstance(file_path, str):
        raise ValueError("file_path must be a non-empty string")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    
    start_time = time.time()
    print(f"Subject Data:  \n{subject_data}")
    
    try:
        # Step 1: Extract chapters first (fast - only processes first 10 pages)
        logger.info("Step 1: Extracting chapters...")
        chapters = extract_chapters(file_path, num_pages=max_toc_pages, max_workers=10)
        logger.info(f"Extracted {len(chapters)} chapters")
        
        # Step 2: Extract content from ALL pages
        logger.info("Step 2: Extracting content from ALL pages...")
        elements = extract_multimodal_elements_from_pdf(file_path, max_workers=60)  # Increased for maximum parallelism
        logger.info(f"Extracted {len(elements)} elements")

        # Step 3: Create chunks
        logger.info("Step 3: Creating chunks...")
        chunks = create_smart_chunks(subject_data, elements, chapters)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Total processing time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        logger.info(f"Created {len(chunks)} chunks from {len(chapters)} chapters")
        
        return chapters, chunks
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Error processing PDF after {elapsed_time:.2f} seconds: {e}", exc_info=True)
        raise

# ---- Utility Functions For Retrieving ----
def get_retriever_by_topic(topic_numbers: List[str], topic_names: List[str], request_data: dict, k: int = 10):
    """
    Retrieve Chunks using Strict Filtering by PDF Name and Chapter.
    """
    if not os.path.exists(INDEX_DIR):
        raise FileNotFoundError(f"FAISS index not found at {INDEX_DIR}")
    
    try:
        vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        
        # --- STRICT FILTER FUNCTION ---
        def strict_filter(metadata):
            # 1. CRITICAL: Filter by Unique PDF Name
            # This ensures we only get chunks from the specific book requested
            requested_pdf = request_data.get("pdf_name")
            if requested_pdf:
                # Compare exactly. If metadata doesn't have pdf_name, skip it.
                if metadata.get("pdf_name") != requested_pdf:
                    return False
            
            # 2. Filter by Chapter Number
            # We only want chunks from the chapters selected in the UI
            doc_chapter = str(metadata.get("chapter_no", ""))
            allowed_chapters = [str(t) for t in topic_numbers]
            
            if doc_chapter not in allowed_chapters:
                return False
            
            return True
        # ------------------------------

        # INCREASE K to ensure we get "ALL" chunks for the topics
        # If the user wants 3 chapters, we want deeper retrieval than just top-10 global.
        # We increase 'k' significantly because the filter will prune unrelated docs anyway.
        effective_k = k * 3  # Fetch more, then filter down

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": effective_k, 
                "filter": strict_filter # Apply the strict filter
            }
        )
        return retriever

    except Exception as e:
        logger.error(f"Error loading retriever: {e}", exc_info=True)
        raise

def get_prompt_template(subject: str, requested_type: str):
    subject = subject.lower()
    requested_type = requested_type.lower()

    # -------- FORMAT RULES --------
    # Add this global rule or append it to specific sections
    latex_instruction = """
    - *LATEX REQUIREMENT:* If the answer involves numbers, equations, or units, the correct_answer field MUST be formatted in LaTeX (enclosed in $...$).
      Example: "$x = 5$" instead of "x = 5".
    """

    # -------- FORMAT RULES --------
    if any(x in requested_type for x in ['mcq', 'multiple choice', 'objective']):
        format_rules = f"""
        **FORMAT: MULTIPLE CHOICE (STRICT)**
        - Exactly 4 options: (a), (b), (c), (d).
        - Only ONE correct option.
        - No vague or overlapping options.
        - Options must be age-appropriate for the given class.
        - `correct_answer` must EXACTLY match one option.
        - + {latex_instruction}  
        """
    elif any(x in requested_type for x in ['true', 'false']):
        format_rules = """
        **FORMAT: TRUE / FALSE**
        - Only two options: True, False.
        - Statement must be clearly verifiable.
        - Avoid trick or ambiguous language.
        """
    elif any(x in requested_type for x in ['fill', 'blank']):
        format_rules = """
        **FORMAT: FILL IN THE BLANKS**
        - Only ONE blank per question.
        - No options.
        - Blank must test a key concept or calculation.
        """
    else:
        format_rules = """
        **FORMAT: SUBJECTIVE / DESCRIPTIVE**
        - No options.
        - Questions must clearly specify what is expected.
        - `correct_answer` should include key points or final answer.
        """

    # -------- SUBJECT RULES --------
    if subject in ['maths', 'mathematics']:
        subject_rules = """
        ### MATHEMATICS RULES (NON-NEGOTIABLE)
        1. NO theory-based questions.
        2. Every question must be solvable numerically or logically.
        3. Do NOT ask definitions or formulas directly.
        4. Questions must be self-contained.
        5. Difficulty must match the given class level.
        6. Use $$LaTeX$$ for all mathematical expressions.
        """
    elif subject in ['science', 'physics', 'chemistry', 'biology']:
        subject_rules = """
        ### SCIENCE RULES
        1. Follow NCERT / State Board phrasing.
        2. Avoid experimental ambiguity.
        3. Diagrams should be implied, not referenced.
        4. Higher classes may include reasoning-based questions.
        """
    elif subject in ['english', 'hindi']:
        subject_rules = """
        ### LANGUAGE RULES
        - Strictly follow the textbook context provided.
        - Ensure questions test specific language skills (Reading, Writing, Grammar).
        - Do NOT hallucinate poems or stories not in the context.
        """
        # 1. PROSE / COMPREHENSION SECTIONS
        if any(x in requested_type for x in ['read the extract','prose', 'comprehension', 'passage', 'extract']):
            format_rules = """
            **FORMAT: READING COMPREHENSION (BOARD PATTERN)**
            1. **MANDATORY:** You MUST extract a full paragraph (15-20 lines) from the Context and put it in the `description` field of the JSON output.
            2. **QUESTIONS:** Generate 'Activities' based *only* on that passage:
               - **A1 (Simple Factual):** True/False, Fill in blanks, or Complete sentences.
               - **A2 (Complex Factual):** Web diagram, Complete the table, or Arrange in sequence. *Describe the diagram structure in text.*
               - **A3 (Vocabulary):** Find synonyms/antonyms/phrases from passage.
               - **A4 (Grammar):** Do as directed based on sentences in the passage.
               - **A5 (Personal Response):** Open-ended question related to the passage theme.
            3. **NO MCQs.**
            """
            subject_rules = """
            - The passage MUST be verbatim from the provided context.
            - Questions must follow the standard 10th Board Activity Sheet pattern.
            """

        # 2. POETRY SECTIONS
        elif any(x in requested_type for x in ['poem', 'poetry', 'appreciation']):
            format_rules = """
            **FORMAT: POETRY COMPREHENSION**
            1. **MANDATORY:** Extract the full poem (or stanzas) from the Context and put it in the `description` field.
            2. **GENERATE ACTIVITIES:**
               - A1: Simple factual (pick out lines, true/false).
               - A2: Explain lines / Poetic Device / Rhyme Scheme.
               - A3: Appreciation / Theme of the poem.
            3. **NO MCQs.**

            **FORMAT: APPRECIATION OF POEM**
            - Generate a single long question or a set of points (Title, Poet, Rhyme Scheme, Theme).
            - The `description` should contain the Poem Title being appreciated.
            """
            subject_rules += "\n- If the poem text is not in the context, return an error in the description rather than inventing one."

        # 3. GRAMMAR / DO AS DIRECTED
        elif any(x in requested_type for x in ['grammar', 'do as directed', 'language study']):
            format_rules = """
            **FORMAT: LANGUAGE STUDY (GRAMMAR)**
            - Generate standalone grammar questions (Voice, Speech, Transformation, Spot error).
            - For each question, provide the sentence and the specific instruction (e.g., "Change to Passive Voice").
            - `correct_answer` must contain the rewritten correct sentence.
            - **NO MCQs.**
            """

        # 4. WRITING SKILLS
        elif any(x in requested_type for x in ['writing', 'letter', 'report', 'speech', 'expansion']):
            format_rules = """
            **FORMAT: WRITING SKILLS**
            - Generate Topics based on the themes in the context (e.g., "Write a letter to...", "Draft a speech on...").
            - In the `question` field, provide the prompt and any hints/points to use.
            - `correct_answer` should be a "Key Points" checklist for evaluation.
            """
            
    else:
        subject_rules = """
        ### GENERAL ACADEMIC RULES
        - Stick strictly to textbook facts.
        - Avoid dates/numbers not present in context.
        """

    # -------- FINAL PROMPT --------
    return PromptTemplate(
    input_variables=[
        "subject",
        "context",
        "request_data",
        "question_type",
        "num_of_questions",
        "marks_per_question",
        "class_level",
        "format_instructions"
    ],
    template="""
You are an experienced board-level examiner preparing a **final examination question paper**.

### EXAM DETAILS
- Subject: {subject}
- Class: {class_level}
- Question Type: {question_type}
- Number of Questions: {num_of_questions}
- Marks per Question: {marks_per_question}

### STRICT INSTRUCTIONS
1. Questions must be **100% syllabus-aligned**.
2. Difficulty must strictly match **Class {{class_level}}**.
3. No repetition of questions or patterns.
4. No references to textbook pages, figures, or examples.
5. No meta explanations.
6. Language must be formal and exam-oriented.

### QUESTION DESIGN
- Mix direct, application, and reasoning questions.
- For Multiple Choice Questions Or Single Correct Questions:
   - Create exactly 4 distinct and plausible options which can be derived from the TEXTBOOK CONTENT for every question.
   - Ensure only one correct answer.
   - The "correct answer" field must contain the full correct option text (e.g., "a) ..."). 
- Clearly mention the chapter number from which each question is generated or selected.. 
- Avoid unnecessary complexity for lower classes.
- Ensure fairness and clarity.

### FORMAT RULES
""" + format_rules + """

### SUBJECT RULES
""" + subject_rules + """

### INPUT FROM USER
{request_data}

### TEXTBOOK CONTEXT (AUTHORITATIVE)
{context}

### OUTPUT FORMAT (MANDATORY)
{format_instructions}

SCHEMA TO FOLLOW:
{{
    "sectionTitle": "", // question_type
    "description": "",  // Description about question_type
    "questions": [
      {{
        "questionNo": "",
        "question": "",
        "options": [], // Only for MCQs
        "marks": 0,
        "correct_answer": "",
        "chapterNo": 0
      }}
    ]
}}

Return ONLY valid JSON.
"""
)


#     return PromptTemplate(
#         input_variables=[
#             "context",
#             "request_data",
#             "question_type",
#             "num_of_questions",
#             "marks_per_question",
#             "class_level",
#             "format_instructions"
#         ],
#         template=f"""
# You are an experienced board-level examiner preparing a **final examination question paper**.

# ### EXAM DETAILS
# - Subject: {subject}
# - Class: {{class_level}}
# - Question Type: {{question_type}}
# - Number of Questions: {{num_of_questions}}
# - Marks per Question: {{marks_per_question}}

# ### STRICT INSTRUCTIONS
# 1. Questions must be **100% syllabus-aligned**.
# 2. Difficulty must strictly match **Class {{class_level}}**.
# 3. No repetition of questions or patterns.
# 4. No references to textbook pages, figures, or examples.
# 5. No meta explanations.
# 6. Language must be formal and exam-oriented.

# ### QUESTION DESIGN
# - Mix direct, application, and reasoning questions.
# - Avoid unnecessary complexity for lower classes.
# - Ensure fairness and clarity.

# ### FORMAT RULES
# {format_rules}

# ### SUBJECT RULES
# {subject_rules}

# ### INPUT FROM USER
# {{request_data}}

# ### TEXTBOOK CONTEXT (AUTHORITATIVE)
# {{context}}

# ### OUTPUT FORMAT (MANDATORY)
# {{format_instructions}}

# SCHEMA TO FOLLOW:
# {{
#     "sectionTitle": "", // question_type
#     "description": "",  // Description about question_type
#     "questions": [
#       {{
#         "questionNo": "",
#         "question": "",
#         "options": [],  // options should be only for Multiple Choice Questions (MCQs)
#         "marks": 0,
#         "correct_answer": ""
#       }}
#     ]
# }}
# Now return ONLY the completed JSON following the schema. 
# REMEMBER: Only raw JSON, no extra text.
# Return ONLY valid JSON.
# """
#     )


def get_context_from_request(request_data: dict, k: int = 15): 
    """
    IMPROVED: Robust Chapter Matching & MMR Retrieval.
    """
    if not request_data or "questions" not in request_data:
        raise ValueError("Invalid request data")
    
    logger.info(f"Generating exam using MMR (Diversity Search)...")
    
    exam_paper = {
        "subject": request_data.get("subject", ""),
        "className": request_data.get("class", ""),
        "maxMarks": request_data.get("maxMarks", 0),
        "timeAllowed": request_data.get("timeAllowed", ""),
        "instructions": request_data.get("instructions", []),
        "sections": []
    }

    if not os.path.exists(INDEX_DIR): 
        raise FileNotFoundError(f"Index not found at {INDEX_DIR}")
        
    vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

    # 1. Determine Subject to tune retrieval parameters
    subject = request_data.get("subject", "").lower()
    is_language = subject in ['english', 'hindi', 'marathi', 'sanskrit']

    for q_idx, q in enumerate(request_data["questions"]):
        q_type = q.get("type", "Unknown")
        topics = q.get("topics", [])
        
        target_chapters = [] 
        for t in topics:
            match = re.search(r"(\d+)", str(t))
            if match:
                chap_id = match.group(1)
                chap_name = str(t)
                target_chapters.append((chap_id, chap_name))
            else:
                target_chapters.append((str(t), str(t)))

        if not target_chapters: 
            logger.warning(f"Could not parse any chapter numbers from topics: {topics}")
            continue

        all_contexts = []
        
        # 2. INCREASE CHUNK SIZE FOR LANGUAGES
        # Language papers need full stories/poems (large context). 
        # Math/Science need specific formulas (small context).
        base_k = 30 if is_language else 15
        chunks_per_chapter = max(5, base_k // len(target_chapters))

        for chap_id, chap_name in target_chapters:
            
            def specific_chapter_filter(metadata):
                # 1. Check PDF Name
                if request_data.get("pdf_name"):
                    if metadata.get("pdf_name") != request_data.get("pdf_name"):
                        return False
                
                # 2. Check Chapter Number (Robust String/Int Match)
                meta_chap = str(metadata.get("chapter_no", "")).strip()
                target_chap = str(chap_id).strip()
                
                if meta_chap == target_chap: return True
                if meta_chap.lstrip("0") == target_chap.lstrip("0"): return True
                return False

            try:
                retriever = vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": chunks_per_chapter, 
                        "fetch_k": chunks_per_chapter * 5,  # Fetch more candidates
                        "lambda_mult": 0.7, 
                        "filter": specific_chapter_filter
                    }
                )
                
                # Search query
                query = f"{chap_name} {q.get('llm_note', '')}"
                docs = retriever.invoke(query)
                
                if docs:
                    logger.info(f"Retrieved {len(docs)} chunks for Chapter {chap_id}")
                    for d in docs:
                        all_contexts.append(f"[Source: Chapter {chap_id}]\n{d.page_content}")
                else:
                    logger.warning(f"Zero docs for Chapter {chap_id}. Check metadata or extraction.")

            except Exception as e:
                logger.error(f"Retrieval error for chapter {chap_id}: {e}")

        # 3. DISABLE GLOBAL FALLBACK
        # Do NOT search globally if chapter filters fail. This prevents "Chapter 4" 
        # from appearing when you asked for "Chapter 7".
        if not all_contexts:
            logger.error(f"CRITICAL: No valid content found for {q_type} in selected chapters.")
            # Skip this section rather than hallucinating
            continue

        import random
        random.shuffle(all_contexts)
        # Limit total context size to fit in prompt (approx 12k chars)
        combined_context = "\n\n".join(all_contexts)[:25000] 

        # 4. Generate with DYNAMIC PROMPT
        try:
            print(f"Generating Section: {q_type}")
            
            selected_prompt = get_prompt_template(subject, q_type)
            
            marks_val = q.get("marks") or q.get("marksPerQuestion") or 1
            
            final_prompt = selected_prompt.format(
                subject=subject,
                context=combined_context,
                request_data=json.dumps(q, indent=2),
                question_type=q_type,
                num_of_questions=q.get("numQuestions", 0),
                class_level=request_data.get("class", "Unknown"),
                marks_per_question=marks_val,
                format_instructions=parser.get_format_instructions()
            )
            
            response = llm.invoke(final_prompt)
            cleaned_text = clean_json_string(response.content)
            parsed = json.loads(cleaned_text)
            
            if isinstance(parsed, dict):
                exam_paper["sections"].append({
                    "sectionName": q.get("sectionName", ""),
                    "sectionTitle": parsed.get("sectionTitle", ""),
                    "description": parsed.get("description", ""),
                    "questions": parsed.get("questions", [])
                })
        except Exception as e:
            logger.error(f"Error generating section {q_type}: {e}")

    return exam_paper

def summarize_questions(llm, questions):
    """
    Summarize a list of questions into short conceptual summaries.
    
    Args:
        llm: LLM instance
        questions: List of question strings
    
    Returns:
        List of summary strings
    """
    if not questions:
        logger.debug("No questions provided for summarization")
        return []
    
    if not llm:
        logger.error("LLM instance is None")
        return []

    try:
        prompt = f"""
        Summarize the following questions into short one-line conceptual summaries 
        that capture what each question is testing (without copying exact phrasing).

        Questions:
        {json.dumps(questions, indent=2)}

        Return as a bullet list, no numbering, no extra text.
        """

        resp = llm.invoke(prompt)
        if not resp or not hasattr(resp, 'content'):
            logger.warning("Invalid LLM response for question summarization")
            return []
        
        lines = [line.strip("-• ").strip() for line in resp.content.split("\n") if line.strip()]
        logger.info(f"Generated {len(lines)} question summaries")
        return lines
    except Exception as e:
        logger.error(f"Summary generation failed: {e}", exc_info=True)
        return []

def generate_multiple_papers_with_summaries(request_data: dict, num_papers: int = 1, k: int = 10):
    """
    Generate multiple diverse question papers, using concept summaries to avoid repetition.
    
    Args:
        request_data: Dictionary containing exam request data
        num_papers: Number of papers to generate
        k: Number of chunks to retrieve
    
    Returns:
        List of exam paper dictionaries
    
    Raises:
        ValueError: If inputs are invalid
    """
    if not request_data or not isinstance(request_data, dict):
        raise ValueError("request_data must be a non-empty dictionary")
    
    if num_papers <= 0:
        raise ValueError("num_papers must be greater than 0")
    
    if k <= 0:
        raise ValueError("k must be greater than 0")
    
    logger.info(f"Generating {num_papers} diverse question papers")
    
    generated_papers = []
    previous_concept_summaries = []

    for paper_no in range(num_papers):
        try:
            logger.info(f"Generating Paper {paper_no + 1}/{num_papers}")

            # Create diversity hint
            if previous_concept_summaries:
                diversity_hint = (
                    "Avoid creating questions similar to these concepts:\n"
                    + "\n".join(previous_concept_summaries[-40:])
                )
            else:
                diversity_hint = "Create unique and diverse questions covering all given topics."

            # Clone request safely
            try:
                modified_request = deepcopy(request_data)
            except Exception as e:
                logger.error(f"Error cloning request data: {e}", exc_info=True)
                raise
            
            if "questions" not in modified_request:
                raise ValueError("request_data must contain 'questions' key")
            
            for q in modified_request["questions"]:
                if not isinstance(q, dict):
                    continue
                if "llm_note" in q and isinstance(q["llm_note"], list):
                    q["llm_note"].append(diversity_hint)
                else:
                    q["llm_note"] = [diversity_hint]

            # Generate one paper using your existing function
            try:
                paper = get_context_from_request(modified_request, k=k)
                if not paper or not isinstance(paper, dict):
                    logger.warning(f"Invalid paper generated for paper {paper_no + 1}")
                    continue
                generated_papers.append(paper)
            except Exception as e:
                logger.error(f"Error generating paper {paper_no + 1}: {e}", exc_info=True)
                continue

            # Extract questions for summarization
            all_questions = []
            for sec in paper.get("sections", []):
                if not isinstance(sec, dict):
                    continue
                for ques in sec.get("questions", []):
                    if isinstance(ques, dict):
                        question_text = ques.get("question", "")
                        if question_text:
                            all_questions.append(question_text)

            # Summarize to concept-level
            if all_questions:
                new_summaries = summarize_questions(llm, all_questions)
                previous_concept_summaries.extend(new_summaries)
                previous_concept_summaries = previous_concept_summaries[-100:]  # keep rolling window
        except Exception as e:
            logger.error(f"Unexpected error generating paper {paper_no + 1}: {e}", exc_info=True)
            continue

    logger.info(f"Successfully generated {len(generated_papers)}/{num_papers} papers")
    return generated_papers


# ---------- ✳️ ANSWER EVALUATION MODULE Functions ✳️ ----------
def extract_contents_from_pdf(file_path: str):
    """
    Extract all handwritten text (answers, names, roll numbers, etc.) from a student's handwritten answer sheet.
    """

    try:
        pages = convert_from_path(pdf_path=file_path) 
    except Exception as e:
        logger.error(f"Error converting PDF to images: {e}")
        # Fallback for Windows local dev if needed, or re-raise
        raise RuntimeError(f"Could not convert PDF. Is Poppler installed? Error: {e}")

    #model = genai.GenerativeModel(model_name="gemini-2.5-flash")
    model = GenerativeModel("gemini-2.5-flash")

    # ✨ Carefully crafted prompt
    prompt = """
You are an OCR and handwriting recognition expert. 
You are given a scanned page from a student's handwritten answer sheet. 
Your goal is to extract **only** the handwritten content written by the student with maximum accuracy. 

### Extraction Rules:
1. Extract everything written by hand — including Name, Roll Number, Class, Subject, Question Numbers, and all Answers.
2. Preserve the **exact words, spellings, mathematical notations, symbols, and diagrams descriptions** as visible.
3. Do **not** skip crossed-out text; mention it as: (crossed out: "<text>")
4. Do **not** interpret or summarize — just transcribe exactly what is written.
5. Maintain natural line breaks and structure (use `\n` where new lines are visible).
6. Ignore printed templates, page numbers, logos, headers, or margins.
7. If handwriting is unclear or ambiguous, mark it as `[unclear]`.

### Output format:
Return plain text, no Markdown, no code blocks.
The format should look like:

------------------------------
Page 1 Text:
<exact handwritten transcription>

If multiple pages are provided, continue as:
------------------------------
Page 2 Text:
<text>
------------------------------
    """

    extracted_text = ""
    for page_num, page in enumerate(pages, start=1):
        try:
            # FIX: Use safe_vertex_generate instead of model.generate_content
            # This uses the new retry logic we just added
            response = safe_vertex_generate(model, [prompt, pil_to_part(page)])
            
            if response and response.candidates and response.candidates[0].content.parts:
                content = response.candidates[0].content.parts[0].text.strip()

                if content.startswith("```"):
                    content = content.strip("`").replace("json", "").strip()

                extracted_text += f"\n\n--- Page {page_num} ---\n{content}"
            else:
                logger.warning(f"Empty response for page {page_num}")

            # FIX: Add a small sleep to prevent hitting rate limits
            time.sleep(2) 

        except Exception as e:
            logger.error(f"Error processing page {page_num}: {e}")

    logger.info(f"Extracted {len(extracted_text)} characters from answer sheet.")
    print(f"Extracted Text from the pdf: \n{extracted_text}")
    return extracted_text


def assign_marks(question: str, correct_answer: str, student_answer: str, max_marks: int):
    """
    Uses Vertex AI (Gemini) to evaluate the student's handwritten answer and assign marks.
    """

    prompt = f"""
You are a strict, rule-bound examiner evaluating a student's handwritten answer sheet.

Question: {question}
Correct Answer: {correct_answer}
Student's Handwritten Answer: {student_answer}

---

### 🔥 OVERALL EVALUATION PRINCIPLE:
Award marks **only for what is explicitly written by the student**, not for what they “might have meant”.  
No assumptions. No generosity. No filling gaps.

---

## 🎯 MARKING RULES (STRICTER VERSION)

Note : if the student answer is Empty or Blank ( "" ) , award 0 marks. [ VERY VERY IMPORTANT ]

### 1. Zero-tolerance for missing required points  
- If the question demands **specific items** (e.g., “herders, farmers, merchants, kings”), the answer MUST explicitly mention them.  
- If even one required item is missing → deduct marks proportionally.

### 2. Blank / Irrelevant / Incorrect → **0 marks**
- Even partially related but off-target content = 0.
- If the student answers only half the question (e.g., only definition, no explanation) → heavy deductions.

### 3. No reward for general knowledge  
- Marks only for content aligning with the **correct answer or textbook context**.  
- Irrelevant extra information → **no marks**.

### 4. Partial marks only for:
- Clearly correct points **directly answering the question**.
- Each required point contributes a **fixed fraction** of the marks.
- Vague statements that don’t show clear understanding → **0**.

### 5. Specificity Required  
- General statements like “people lived differently” or “past was different for everyone” are NOT enough for 3+ mark questions.
- Examples, categories, and named items MUST appear for credit.

### 6. Structure Matters (for long answers)
Marks deducted for:
- Missing introduction
- Missing explanation
- Missing required examples
- No logical flow

### 7. Factual accuracy required  
- Wrong facts → zero marks for that portion.
- Spelling errors that change meaning → deduct marks.
- Minor spelling mistakes that do not change meaning → do not award but do not penalize heavily.

### 8. No marks for repetition or fluff  
- Rewriting the question in different words earns **no credit**.

---

## 📌 MARKING GUIDE BY QUESTION TYPE (STRICT)

### 🔸 **MCQ / One-word / Fill-in-the-Blank**
- **Primary Check:** Does the **Option Letter** (a, b, c, d) match? If yes → Full Marks.
- **Secondary Check:** Does the **Content** match? 
  - **IGNORE formatting differences** (e.g., "$120~cm^2$" vs "120 cm²", "x" vs "×").
  - If the value/meaning is identical → Full Marks.
- Otherwise → 0 Marks.

---

### 🔸 **Short Answer (1–3 marks)**
Award marks ONLY if:
- The **key phrase/term** is exactly present.
- The explanation matches the correct answer.

Penalize:
- Missing keywords
- Vague explanation
- Off-topic examples
- Incorrect definitions

---

### 🔸 **Medium-length / Reasoning (3–4 marks)**
Expect:
- Clear definition + required explanation
- All required key points

Deductions for:
- Missing examples
- Missing second part of question
- Partial conceptual understanding
- Incomplete comparisons

---

### 🔸 **Long Answer / Analytical (5–6 marks)**
Expect:
- Intro / definition
- Explanation
- Examples or points explicitly present
- All subparts answered

If ANY required component missing:
- Deduct 1–2 marks immediately.

If multiple missing:
- Award very low marks or 0.

---

### ⭐ **MARK DISTRIBUTION RULE (very strict):**
For multi-point questions:
- Each correct explicit point = (max_marks / number_of_required_points)
- Missing point = 0 for that portion.
- Vague/generalised point = 0.

---

## ✔️ FINAL OUTPUT FORMAT
You MUST return ONLY valid JSON. No markdown, no code blocks, no explanations outside the JSON.

The maximum marks for this question is {max_marks}.

Return ONLY this JSON format (nothing else):

{{
  "awarded_marks": <number between 0 and {max_marks}>,
  "remarks": "<brief strict justification>"
}}

IMPORTANT: Return ONLY the JSON object, no other text before or after it.
"""

    try:
        # Use Vertex AI instead of Groq
        model = GenerativeModel("gemini-2.5-flash")
        response = safe_vertex_generate(model, [prompt])
        
        if not response:
            logger.warning("No response from Vertex AI for mark assignment")
            return {"awarded_marks": 0, "remarks": "Error: No response from AI"}
        
        if not response.candidates or len(response.candidates) == 0:
            logger.warning("No candidates in Vertex AI response")
            return {"awarded_marks": 0, "remarks": "Error: No candidates in response"}
        
        # Note: We'll check for content existence rather than finish_reason
        # as finish_reason enum values may vary between Vertex AI versions
        
        if not response.candidates[0].content.parts or len(response.candidates[0].content.parts) == 0:
            logger.warning("No content parts in Vertex AI response")
            return {"awarded_marks": 0, "remarks": "Error: No content parts in response"}
        
        raw = response.candidates[0].content.parts[0].text
        if not raw:
            logger.warning("Empty text in Vertex AI response")
            return {"awarded_marks": 0, "remarks": "Error: Empty text in response"}
        
        raw = raw.strip()
        logger.info(f"Raw Evaluation Output (first 500 chars): {raw[:500]}")

        # Try to parse JSON directly first
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "awarded_marks" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass
        
        # Handle cases where model adds extra text or markdown
        cleaned = raw
        
        # Remove markdown code blocks if present
        if "```json" in cleaned:
            cleaned = re.sub(r'```json\s*', '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'```\s*$', '', cleaned, flags=re.MULTILINE)
        elif "```" in cleaned:
            cleaned = re.sub(r'```\s*', '', cleaned)
        
        # Remove any leading/trailing whitespace
        cleaned = cleaned.strip()
        
        # Try to extract JSON object using regex
        json_match = re.search(r'\{[\s\S]*\}', cleaned)
        if json_match:
            cleaned = json_match.group(0)
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict) and "awarded_marks" in parsed:
                    return parsed
            except json.JSONDecodeError as e:
                logger.debug(f"JSON decode error after regex extraction: {e}")
                # Try to use clean_json_string helper
                try:
                    cleaned_json = clean_json_string(cleaned)
                    parsed = json.loads(cleaned_json)
                    if isinstance(parsed, dict) and "awarded_marks" in parsed:
                        return parsed
                except Exception as e2:
                    logger.debug(f"clean_json_string also failed: {e2}")
        
        # If still failing, try to find and extract just the JSON part more carefully
        # Look for the pattern: "awarded_marks": number
        marks_match = re.search(r'"awarded_marks"\s*:\s*(\d+)', cleaned)
        remarks_match = re.search(r'"remarks"\s*:\s*"([^"]*)"', cleaned)
        
        if marks_match:
            try:
                awarded_marks = int(marks_match.group(1))
                remarks = remarks_match.group(1) if remarks_match else "Parsed from partial response"
                logger.warning(f"Extracted partial JSON: awarded_marks={awarded_marks}")
                return {"awarded_marks": awarded_marks, "remarks": remarks}
            except:
                pass
        
        # Last resort: log the full response for debugging
        logger.error(f"Failed to parse JSON from response. Full response (first 1000 chars): {raw[:1000]}")
        logger.error(f"Cleaned response (first 1000 chars): {cleaned[:1000]}")
        return {"awarded_marks": 0, "remarks": f"Error: Could not parse JSON from AI response. Response length: {len(raw)} chars"}
            
    except Exception as e:
        logger.error(f"Error in assign_marks: {e}", exc_info=True)
        return {"awarded_marks": 0, "remarks": f"Error: {str(e)}"}

def retrieve_section_answers(section_title: str, questions: list, answer_paper: str):
    """
    Uses Vertex AI (Gemini) once per section to extract answers for all questions.
    """
    # Create a summary of what we are looking for to help the AI Contextualize
    # e.g., "Q1 (MCQ), Q2 (Factorization)..."
    q_descriptions = []
    for q in questions:
        q_snippet = q['question'][:50] + "..." if len(q['question']) > 50 else q['question']
        q_descriptions.append(f"Q{q['questionNo']}: {q_snippet}")
    
    questions_summary = "\n".join(q_descriptions)

    # Note the double curly braces {{ }} around LaTeX commands inside the f-string
    prompt = f"""
You are an expert exam evaluator extracting answers from a student's handwritten script.

### CONTEXT
The student's answer sheet contains multiple sections (e.g., Section A, Section B, or Section 1, 2).
Question numbers (Q1, Q2...) **repeat** in every section.

### YOUR GOAL
Extract the answers ONLY for the specific section described below.
**Section Title:** "{section_title}"
**Questions to find:**
{questions_summary}

### HANDWRITTEN CONTENT
---
{answer_paper}
---

### EXTRACTION RULES
1. **Locate the Section:** Scan the handwritten text for headers like "Section 1", "Section A", or look for answers that match the *content* of the questions listed above.
   - *Example:* If the questions are MCQs, look for short answers like "a) 4".
   - *Example:* If the questions are "Solve the following", look for long steps.
2. **Handle Duplicate Numbers:** Do NOT extract "Q1" from Section 2 if I am asking for "Q1" from Section 1. Use the context of the answer content to disambiguate.
3. **LaTeX Formatting:** - If the answer contains math, format it as LaTeX enclosed in `$ ... $`.
   - **CRITICAL:** Use DOUBLE backslashes for commands. Example: Write `$60 \\\\text{{ km/h}}$` (not `\\text`).
   - **Multiplication:** Write `$a \\\\times b$` (produces $\times$), NOT `$a \times b$`.
   - **Text:** Write `$60 \\\\text{{ km/h}}$`.
   - **Fractions:** Write `$\\\\frac{{1}}{{2}}$`.
   - If the student wrote "atimesb", correct it to `$a \\\\times b$` if it clearly means multiplication.
4. **Precision:** Extract exactly what is written. Do not correct spelling. If an answer is missing, return an empty string.
5. **PRESERVE NEWLINES (CRITICAL):** - If the answer is written across multiple lines (e.g., steps in a math problem, or points in a theory answer), **you MUST use `\\n`** in the JSON string to represent those line breaks.
   - **DO NOT** flatten the text into a single line.
   - Example: "Step 1: Formula\\nStep 2: Substitution\\nStep 3: Answer"

Return **only valid JSON**:
{{
  "answers": [
    {{ "questionNo": "1", "studentAnswer": "..." }},
    {{ "questionNo": "2", "studentAnswer": "..." }}
  ]
}}
"""

    try:
        # Use Vertex AI
        model = GenerativeModel("gemini-2.5-flash")
        response = safe_vertex_generate(model, [prompt])
        
        if not response or not response.candidates or not response.candidates[0].content.parts:
            logger.warning("Empty response from Vertex AI for answer extraction")
            return {}
        
        text = response.candidates[0].content.parts[0].text.strip()

        # Handle possible cases where LLM wraps JSON in ```json ... ```
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            text = match.group(0)

        # Use clean_json_string to fix any remaining escape issues
        text = clean_json_string(text)
        
        data = json.loads(text)
        cleaned_answers = {}
        for a in data.get("answers", []):
            raw_ans = a.get("studentAnswer", "")
            # APPLY CLEANING HERE
            cleaned_answers[a["questionNo"]] = clean_latex_content(raw_ans)
            
        return cleaned_answers

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON returned by Vertex AI. Full response:\n{text if 'text' in locals() else 'No response'}")
        return {}
    except Exception as e:
        logger.error(f"Error retrieving section answers: {e}", exc_info=True)
        return {}

def analyze_chapters_with_llm(report: dict):
    """
    Call LLM (Vertex AI) to compute chapter-wise totals and produce strengths/weaknesses/recommendations.
    Expects 'report' to contain per-question awarded marks.
    """
    # Build a compact per-question table for the LLM
    per_question_list = []
    for section in report.get("sections", []):
        for q in section.get("questions", []):
            per_question_list.append({
                "questionNo": q.get("questionNo"),
                # Handle both "chapterNo" (your new standard) and "chapter" (legacy/fallback)
                "chapterNo": q.get("chapterNo", q.get("chapter", "Unknown")),
                "marks": q.get("marks", 0),
                "awarded": q.get("awarded", 0),
                "studentAnswer": q.get("studentAnswer", "")
            })

    # Build prompt
    prompt = f"""
    You are an expert exam analyst. You will be given a student's evaluated question-level data.
    
    Your task:
    1) Group questions by 'chapterNo'.
    2) For EACH chapter, calculate:
       - totalMarks: sum of 'marks'
       - obtainedMarks: sum of 'awarded'
       - percentage: (obtainedMarks / totalMarks) * 100
    3) For EACH chapter, provide:
       - strengths: list (max 3) of topics/skills the student answered well.
       - weaknesses: list (max 4) of specific gaps where marks were lost.
       - recommendations: 1-2 specific actionable study tips.
    4) Produce an 'overall_summary' with:
       - strong_chapters: list of chapterNos with high percentages.
       - weak_chapters: list of chapterNos with low percentages.
       - study_plan: list of 3 bullet points for overall improvement.

    Input Data:
    {json.dumps(per_question_list, ensure_ascii=False, indent=2)}

    Output Format (Strict JSON):
    {{
      "chapters": [
        {{
          "chapterNo": "1",
          "totalMarks": 10,
          "obtainedMarks": 8,
          "percentage": 80.0,
          "strengths": ["..."],
          "weaknesses": ["..."],
          "recommendations": "..."
        }}
      ],
      "overall_summary": {{
        "strong_chapters": ["..."],
        "weak_chapters": ["..."],
        "study_plan": ["..."]
      }}
    }}
    """

    try:
        # Use your existing Vertex AI setup
        model = GenerativeModel("gemini-2.5-flash")
        response = safe_vertex_generate(model, [prompt])
        
        if not response or not response.candidates or not response.candidates[0].content.parts:
            logger.warning("No response from Vertex AI for chapter analysis")
            return {}

        text = response.candidates[0].content.parts[0].text.strip()
        cleaned_text = clean_json_string(text)
        return json.loads(cleaned_text)

    except Exception as e:
        logger.error(f"Chapter analysis failed: {e}", exc_info=True)
        # Return empty structure on failure so frontend doesn't break
        return {
            "chapters": [],
            "overall_summary": {
                "strong_chapters": [],
                "weak_chapters": [],
                "study_plan": []
            }
        }

def evaluate_answers(question_paper: dict, answer_paper: str, max_workers: int = 2): 
    """
    Evaluate student's handwritten answers with the official question paper.
    Uses multiprocessing for parallel evaluation of questions.
    
    Args:
        question_paper: Dictionary containing the question paper structure
        answer_paper: Extracted text from student's handwritten answer sheet
        max_workers: Maximum number of parallel workers for evaluation (default: 20)
    
    Returns:
        Dictionary containing evaluation results
    """
    result = []
    total_marks = 0
    obtained_marks = 0

    def evaluate_single_question(q, student_ans):
        """Helper function to evaluate a single question"""
        try:
            logger.info(f"DEBUG Q{q.get('questionNo')}: Keys -> {list(q.keys())}")
            q_text = q['question']
            marks = q.get('marks', 0)
            raw_ans =q.get('correct answer') or q.get('correct_answer') or q.get('correctAnswer')
            correct_ans = clean_latex_content(raw_ans)
            clean_student_ans = clean_latex_content(student_ans)
            # Evaluate marks using Vertex AI
            evaluation = assign_marks(q_text, correct_ans, clean_student_ans, marks)
            
            logger.debug(f"Evaluation for Q{q['questionNo']}: {evaluation}")

            # Parse response safely
            try:
                eval_data = json.loads(evaluation) if isinstance(evaluation, str) else evaluation
                score = eval_data.get("awarded_marks", 0)
                remarks = eval_data.get("remarks", "")
            except Exception as e:
                logger.warning(f"Error parsing evaluation for Q{q['questionNo']}: {e}")
                score = 0
                remarks = "Invalid response format from AI"

            return {
                "questionNo": q['questionNo'],
                "question": q_text,
                "marks": marks,
                "chapterNo": q.get('chapterNo', q.get('chapter', 'Unknown')),# --- CHANGE 1: Capture Chapter No ---
                "studentAnswer": clean_student_ans,
                "correctAnswer": correct_ans,
                "awarded": score,
                "remarks": remarks
            }
        except Exception as e:
            logger.error(f"Error evaluating question {q.get('questionNo', 'unknown')}: {e}", exc_info=True)
            return {
                "questionNo": q.get('questionNo', ''),
                "question": q.get('question', ''),
                "marks": q.get('marks', 0),
                "chapterNo": q.get('chapterNo', 'Unknown'), # Ensure fallback
                "studentAnswer": student_ans,
                "correctAnswer": q.get('correct_answer', ''),
                "awarded": 0,
                "remarks": f"Error during evaluation: {str(e)}"
            }

    for section in question_paper['sections']:
        # FIX: Robustly get the title. Try 'sectionTitle', then 'sectionName', then 'title', then default.
        sec_title = section.get('sectionTitle') or section.get('sectionName') or section.get('title') or "Untitled Section"
        
        section_result = {"sectionTitle": sec_title, "questions": []}

        # ✅ Retrieve all answers for this section in one single LLM call
        answers_map = retrieve_section_answers(
            sec_title, section.get('questions', []), answer_paper
        )

        logger.info(f"Answers Map for section '{sec_title}': {answers_map}")

        # ✅ Evaluate questions in parallel using ThreadPoolExecutor
        questions_to_evaluate = []
        for q in section.get('questions',[]):
            student_ans = answers_map.get(q['questionNo'], "")
            questions_to_evaluate.append((q, student_ans))
            total_marks += q.get('marks', 0)

        # Process questions in parallel
        if questions_to_evaluate:
            try:
                effective_workers = min(max_workers, len(questions_to_evaluate))
                with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as executor:
                    # Submit all evaluation tasks
                    futures = {
                        executor.submit(evaluate_single_question, q, student_ans): q
                        for q, student_ans in questions_to_evaluate
                    }
                    
                    # Collect results as they complete
                    evaluated_questions = []
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            question_result = future.result()
                            evaluated_questions.append(question_result)
                            obtained_marks += question_result.get("awarded", 0)
                        except Exception as e:
                            q = futures.get(future, {})
                            logger.error(f"Error in future result for question {q.get('questionNo', 'unknown')}: {e}", exc_info=True)
                    
                    # Sort by question number to maintain order
                    evaluated_questions.sort(key=lambda x: x.get("questionNo", ""))
                    section_result['questions'] = evaluated_questions
                    
            except Exception as e:
                logger.error(f"Error in parallel evaluation for section '{section['sectionTitle']}': {e}", exc_info=True)
                # Fallback to sequential processing
                for q, student_ans in questions_to_evaluate:
                    question_result = evaluate_single_question(q, student_ans)
                    section_result['questions'].append(question_result)
                    obtained_marks += question_result.get("awarded", 0)

        result.append(section_result)

    #logger.info(f"Evaluation complete: {obtained_marks}/{total_marks} marks")
    #return {
    #    "Subject": question_paper["subject"],
    #   "Class": question_paper["className"],
    #    "totalMarks": total_marks,
    #    "obtainedMarks": obtained_marks,
    #    "sections": result
    #}
    # Base Report
    final_report = {
        "Subject": question_paper["subject"],
        "Class": question_paper["className"],
        "totalMarks": total_marks,
        "obtainedMarks": obtained_marks,
        "sections": result
    }

    # --- CHANGE 2: Run Chapter Analysis ---
    logger.info("Running chapter-wise analysis...")
    chapter_summary = analyze_chapters_with_llm(final_report)
    final_report["chapter_summary"] = chapter_summary
    # --------------------------------------

    logger.info(f"Evaluation complete: {obtained_marks}/{total_marks} marks")
    return final_report


# ---- Init LLM ----
def get_llm():
    """Initialize and return LLM instance"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    
    try:
        logger.info("Initializing ChatGroq LLM")
        llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)
        logger.info("LLM initialized successfully")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
        raise

try:
    llm = get_llm()
    parser = JsonOutputParser()
except Exception as e:
    logger.critical(f"Failed to initialize LLM or parser: {e}", exc_info=True)
    raise

QUESTION_PROMPT = PromptTemplate(
   input_variables=["context", "request_data", "question_type", "num_of_questions", "format_instructions"],
   template="""
You are a senior academic examiner. Generate a high-quality exam section based strictly on the provided textbook context.

### INPUT DATA
**Question Type:** {question_type}
**Number of Questions:** {num_of_questions}
**Context:** Contains extracts from specific chapters.

### STRICT GENERATION RULES

1. **CONTENT INTEGRITY:**
   - Use ONLY information from the [Source: Chapter X] blocks provided.
   - Do not use outside knowledge.
   - Ensure questions are distributed across ALL chapters present in the context.

2. **QUESTION QUALITY (Bloom's Taxonomy):**
   - **Knowledge (30%):** Recall facts, definitions, formulas (e.g., "Define...", "State the formula...").
   - **Understanding (40%):** Explain concepts (e.g., "Explain why...", "Distinguish between...").
   - **Application (30%):** Solve problems or analyze diagrams (e.g., "Calculate...", "Based on the diagram...").
   - *For Math:* Do not ask "What is the formula?". Give values and ask to *solve* using the formula.

3. **FORMATTING RULES (CRITICAL):**
   - **IF MCQ/Objective:**
     - Must provide `options` array with 4 distinct choices.
     - `correct_answer` must match one option text exactly.
   - **IF Subjective (Short/Long/Explain):**
     - `options` array MUST be empty `[]`.
     - `correct_answer` should be the model answer key (bullet points).
   - `chapterNo` MUST be extracted from the `[Source: Chapter X]` tag in the context.

### JSON OUTPUT SCHEMA
{format_instructions}

### SPECIFIC REQUEST DETAILS
{request_data}

### TEXTBOOK CONTEXT
{context}

GENERATE NOW. RETURN ONLY JSON.
"""
)

# Add this helper function after the imports section (around line 90)
def convert_content_to_string(content) -> str:
    """
    Convert content to string format for Document page_content.
    Handles strings, dictionaries (tables), lists, and other types.
    """
    if content is None:
        return ""
    
    if isinstance(content, str):
        return content
    
    if isinstance(content, dict):
        # Handle table structures or other dictionaries
        # Check if it's a table-like structure (has headers key)
        if "headers" in content:
            # It's a table - format it nicely
            headers = content.get("headers", [])
            rows = content.get("rows", [])
            data = content.get("data", [])
            
            # Create table representation
            table_lines = []
            
            # Format headers
            if headers:
                # Convert headers to strings and create header row
                header_strs = [str(h) for h in headers]
                table_lines.append(" | ".join(header_strs))
                # Create separator line
                separator_length = sum(len(h) for h in header_strs) + 3 * (len(header_strs) - 1)
                table_lines.append("-" * max(separator_length, 20))
            
            # Format rows (prefer 'rows' over 'data')
            rows_to_process = rows if rows else data
            if rows_to_process:
                for row in rows_to_process:
                    if isinstance(row, (list, tuple)):
                        # Ensure row has same number of columns as headers
                        row_cells = [str(cell) for cell in row]
                        # Pad or truncate to match header count
                        if headers and len(row_cells) != len(headers):
                            if len(row_cells) < len(headers):
                                row_cells.extend([""] * (len(headers) - len(row_cells)))
                            else:
                                row_cells = row_cells[:len(headers)]
                        table_lines.append(" | ".join(row_cells))
                    elif isinstance(row, dict):
                        # Row is a dictionary - try to extract values in header order
                        if headers:
                            row_values = [str(row.get(h, "")) for h in headers]
                            table_lines.append(" | ".join(row_values))
                        else:
                            table_lines.append(str(row))
                    else:
                        table_lines.append(str(row))
            
            # If no rows but we have headers, still return the table structure
            result = "\n".join(table_lines)
            return result if result.strip() else str(content)
        else:
            # General dictionary - convert to JSON-like string for better readability
            try:
                return json.dumps(content, ensure_ascii=False, indent=2)
            except Exception:
                # Fallback to string representation
                return str(content)
    
    if isinstance(content, (list, tuple)):
        # Convert list to readable format
        # If it's a list of lists, format as table
        if content and all(isinstance(item, (list, tuple)) for item in content):
            # It's a table-like structure (list of rows)
            table_lines = []
            for row in content:
                table_lines.append(" | ".join(str(cell) for cell in row))
            return "\n".join(table_lines)
        else:
            # Regular list - join with newlines
            return "\n".join(str(item) for item in content)
    
    # For any other type, convert to string
    return str(content)

def parse_xml_to_json(xml_text: str) -> dict:
    """
    Parse XML-like tagged blocks to JSON format.
    Handles both single page and multi-page formats.
    """
    if not xml_text:
        return {}
    
    try:
        # Remove any markdown code blocks if present
        xml_text = re.sub(r'```xml\s*', '', xml_text, flags=re.IGNORECASE)
        xml_text = re.sub(r'```\s*', '', xml_text)
        xml_text = xml_text.strip()
        
        # Try to parse as multi-page format first
        pages_pattern = r'<pages>(.*?)</pages>'
        pages_match = re.search(pages_pattern, xml_text, re.DOTALL | re.IGNORECASE)
        
        if pages_match:
            pages_content = pages_match.group(1)
            # Extract all page blocks
            page_pattern = r'<page\s+number=["\']?(\d+)["\']?\s*>(.*?)</page>'
            page_matches = re.finditer(page_pattern, pages_content, re.DOTALL | re.IGNORECASE)
            
            pages_list = []
            for page_match in page_matches:
                page_num = int(page_match.group(1))
                page_content = page_match.group(2)
                
                # Extract elements from this page
                elements = parse_elements_from_xml(page_content)
                if elements:
                    pages_list.append({
                        "page_number": page_num,
                        "elements": elements
                    })
            
            if pages_list:
                return {"pages": pages_list}
        
        # Try single page format (elements)
        elements_pattern = r'<elements>(.*?)</elements>'
        elements_match = re.search(elements_pattern, xml_text, re.DOTALL | re.IGNORECASE)
        
        if elements_match:
            elements_content = elements_match.group(1)
            elements = parse_elements_from_xml(elements_content)
            if elements:
                return {"elements": elements}
        
        # If no structured format found, try to extract any element tags directly
        elements = parse_elements_from_xml(xml_text)
        if elements:
            return {"elements": elements}
        
        return {}
    
    except Exception as e:
        logger.debug(f"Error parsing XML to JSON: {e}")
        return {}

def parse_elements_from_xml(xml_content: str) -> list:
    """
    Extract element tags from XML content.
    Handles CDATA sections and escaped content.
    """
    elements = []
    
    # Pattern to match element tags with optional CDATA
    element_pattern = r'<element\s+type=["\']([^"\']+)["\']\s*>(.*?)</element>'
    element_matches = re.finditer(element_pattern, xml_content, re.DOTALL | re.IGNORECASE)
    
    for match in element_matches:
        try:
            el_type = match.group(1).strip()
            el_content = match.group(2).strip()
            
            # Handle CDATA sections
            if el_content.startswith('<![CDATA[') and el_content.endswith(']]>'):
                el_content = el_content[9:-3]  # Remove CDATA wrapper
            
            # Clean up content (remove extra whitespace, decode entities if needed)
            el_content = re.sub(r'\s+', ' ', el_content).strip()
            
            # Basic HTML/XML entity decoding
            el_content = el_content.replace('&lt;', '<')
            el_content = el_content.replace('&gt;', '>')
            el_content = el_content.replace('&amp;', '&')
            el_content = el_content.replace('&quot;', '"')
            el_content = el_content.replace('&apos;', "'")
            
            if el_type and el_content:
                elements.append({
                    "type": el_type,
                    "content": el_content
                })
        except Exception as e:
            logger.debug(f"Error parsing element: {e}")
            continue
    
    return elements

def clean_json_string(text: str) -> str:
    """Clean JSON string by fixing common escape sequence issues - robust version"""
    if not text:
        return text
    
    # Remove markdown code blocks if present
    text = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```\s*', '', text)
    
    # Process character by character to fix invalid escapes
    # Valid JSON escapes: \" \\ \/ \b \f \n \r \t \uXXXX
    result = []
    i = 0
    
    while i < len(text):
        if text[i] == '\\' and i + 1 < len(text):
            next_char = text[i + 1]
            # Check if it's a valid escape sequence
            if next_char in '"\\/bfnrt':
                # Valid single-character escape - keep as is
                result.append('\\' + next_char)
                i += 2
            elif next_char == 'u':
                # Check for valid unicode escape \uXXXX
                if i + 5 < len(text):
                    hex_part = text[i + 2:i + 6]
                    if len(hex_part) == 4 and all(c in '0123456789abcdefABCDEF' for c in hex_part):
                        # Valid unicode escape - keep as is
                        result.append(text[i:i + 6])
                        i += 6
                    else:
                        # Invalid unicode escape - output \\u
                        result.append('\\\\')
                        result.append('u')
                        i += 2
                else:
                    # Incomplete unicode escape - output \\u
                    result.append('\\\\')
                    result.append('u')
                    i += 2
            else:
                # Invalid escape sequence - output \\ + char
                result.append('\\\\')
                result.append(next_char)
                i += 2
        else:
            result.append(text[i])
            i += 1
    
    # Handle trailing backslash
    if i < len(text) and text[i] == '\\':
        result.append('\\\\')
    
    text = ''.join(result)
    
    # Fix trailing commas before closing braces/brackets
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    # Fix unterminated strings by closing them at the end of JSON structure
    # This handles cases where quotes are missing at the end
    open_quotes = text.count('"') - text.count('\\"')
    if open_quotes % 2 != 0:
        # Odd number of quotes - likely unterminated string
        # Try to find and close the last unclosed string
        # Simple heuristic: if JSON ends without closing quote, add one before the closing brace
        if not text.rstrip().endswith('"') and text.rstrip().endswith('}'):
            # Find the last opening quote that's not closed
            last_quote_pos = text.rfind('"')
            if last_quote_pos > 0:
                # Check if there's content after the last quote
                after_quote = text[last_quote_pos + 1:].strip()
                if after_quote and not after_quote.startswith(','):
                    # Likely unterminated - try to fix by adding quote before closing brace
                    text = re.sub(r'([^"])\s*\}', r'\1"\}', text, count=1)
    
    # Note: Don't auto-fix commas here - let the error handler do it based on specific errors
    # Auto-fixing can break valid JSON
    
    return text

def clean_latex_content(text):
    """
    Cleans text content for proper LaTeX rendering on the frontend.
    1. Removes newlines immediately after equals signs.
    2. Replaces non-standard \degree with ^{\circ}.
    3. Ensures strictly standard spacing.
    4. Merges MCQ option labels with their content (e.g., "a)\nValue" -> "a) Value").
    """
    if not isinstance(text, str):
        return text

    text = re.sub(r'=\s*[\r\n]+\s*', '= ', text)
    text = re.sub(r'\\degree', r'^{\\circ}', text)
    text = re.sub(r'([a-dA-D]\))\s*[\r\n]+\s*', r'\1 ', text)
    return text.strip()

# ---- NEW FUNCTION: Handle Auth and Load Index ----
def initialize_app():
    logger.info("Initializing application...")
    
    # 1. Handle Google Cloud Credentials
    gcp_b64 = os.environ.get("GCP_CREDS_B64")
    if gcp_b64:
        try:
            # Decode the Base64 string
            creds_json = base64.b64decode(gcp_b64)
            # Define a temporary path in /tmp (which is writable in HF)
            creds_path = "/tmp/gcp_creds.json"
            # Write the decoded JSON to the temp file
            with open(creds_path, "wb") as f:
                f.write(creds_json)
            # Set the environment variable for Vertex AI
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
            logger.info("Google Cloud credentials loaded from secret.")
            # Now, re-initialize Vertex AI with the credentials
            vertexai.init(project="gen-lang-client-0238295665", location="us-central1")
            logger.info("Vertex AI re-initialized successfully with secret.")
        except Exception as e:
            logger.error(f"Failed to decode/write GCP credentials or re-init Vertex AI: {e}", exc_info=True)
            raise
    else:
        logger.warning("GCP_CREDS_B64 secret not found. Using local/default credentials.")
        # Try to init again in case it failed the first time (e.g., in local dev)
        try:
            vertexai.init(project="gen-lang-client-0238295665", location="us-central1")
            logger.info("Vertex AI initialized with local/default credentials.")
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI with local/default credentials: {e}")
            raise

    # 2. Handle FAISS Index Loading
    # Check if the index already exists locally
    if os.path.exists(INDEX_DIR):
        logger.info(f"FAISS index found locally at {INDEX_DIR}")
        return

    # If not, try to download it from the HF Dataset
    repo_id = os.environ.get("DATASET_REPO_ID")
    if repo_id:
        logger.info(f"Index not found locally. Attempting to download from HF Dataset: {repo_id}")
        try:
            # Download the entire dataset repo (which contains our index)
            # Note: We download to the root dir '.', so the index appears at './faiss_index'
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir="/tmp",  # Download to the current directory
                allow_patterns="faiss_index/*", # Only download the index folder
                token=os.environ.get("HF_TOKEN") # Add token for private repos
            )
            logger.info(f"Successfully downloaded FAISS index to {INDEX_DIR}")
        except HfHubHTTPError as e:
            # This is okay if it's a 404 (repo is empty or index doesn't exist)
            logger.warning(f"Could not download index (this is normal on first run): {e}")
        except Exception as e:
            logger.error(f"Error downloading index from Hub: {e}")
    else:
        logger.warning("DATASET_REPO_ID not set. Cannot download index.")

# ---- END NEW FUNCTION ----

# ---- CALL THE NEW FUNCTION HERE ----
initialize_app()
# ------------------------------------


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- API Endpoints ----
@app.post("/process_pdf/")
async def process_pdf_endpoint(file: UploadFile=Form(...), subject_data:str = Form(...)):
    """Process PDF and create FAISS index"""
    file_path = None
    try:
        # Validate file
        if not file or not file.filename:
            logger.error("No file provided in request")
            raise HTTPException(status_code=400, detail="No file provided")
        
        if not file.filename.lower().endswith('.pdf'):
            logger.error(f"Invalid file type: {file.filename}")
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        logger.info(f"Processing PDF: {file.filename}")
        
        # Save file locally (use /tmp for writable dir in HF)
        #file_path = f"/tmp/temp_{file.filename}"
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, f"temp_{file.filename}")
        try:
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Uploaded file is empty")
            
            with open(file_path, "wb") as f:
                f.write(content)
            logger.info(f"File saved temporarily: {file_path}")
        except Exception as e:
            logger.error(f"Error saving uploaded file: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

        # Parse subject_data string into dict
        try:
            subject_data_dict = json.loads(subject_data)
            if not isinstance(subject_data_dict, dict):
                raise ValueError("subject_data must be a valid JSON object")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in subject_data: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid JSON in subject_data: {str(e)}")
        except Exception as e:
            logger.error(f"Error parsing subject_data: {e}")
            raise HTTPException(status_code=400, detail=f"Error parsing subject_data: {str(e)}")

        # Process PDF into documents
        try:
            chapters, chunks = process_pdf(subject_data_dict, file_path)
        except FileNotFoundError as e:
            logger.error(f"PDF file not found: {e}")
            raise HTTPException(status_code=404, detail=f"PDF file not found: {str(e)}")
        except ValueError as e:
            logger.error(f"Invalid input: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing PDF: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

        if not chunks:
            logger.warning("No text chunks were created from PDF")
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "No text chunks were created. Likely due to Gemini API rate limits or empty PDF."
                }
            )

        # Create FAISS index
        try:
            logger.info(f"Creating vector store for {len(chunks)} new chunks")
            new_vectorstore = FAISS.from_documents(chunks, embeddings)
            
            # CHECK: Does an index already exist?
            if os.path.exists(INDEX_DIR):
                try:
                    logger.info(f"Loading existing FAISS index from {INDEX_DIR}")
                    existing_vectorstore = FAISS.load_local(
                        INDEX_DIR, 
                        embeddings, 
                        allow_dangerous_deserialization=True
                    )
                    
                    # MERGE the new book into the existing index
                    logger.info("Merging new content into existing index...")
                    existing_vectorstore.merge_from(new_vectorstore)
                    
                    # Save the combined index
                    existing_vectorstore.save_local(INDEX_DIR)
                    logger.info(f"Merged and saved FAISS index to {INDEX_DIR}")
                    
                except Exception as e:
                    logger.error(f"Error merging index: {e}. Falling back to overwriting.")
                    # Fallback: If loading fails, save the new one as the master
                    new_vectorstore.save_local(INDEX_DIR)
            else:
                # No existing index, just save the new one
                logger.info("No existing index found. Creating new one.")
                os.makedirs(INDEX_DIR, exist_ok=True)
                new_vectorstore.save_local(INDEX_DIR)
                logger.info(f"FAISS index saved to {INDEX_DIR}")

            # ---- START NEW CODE: UPLOAD TO HUGGING FACE ----
            repo_id = os.environ.get("DATASET_REPO_ID")
            hf_token = os.environ.get("HF_TOKEN")
            
            if repo_id and hf_token:
                logger.info(f"Attempting to upload index to HF Dataset: {repo_id}")
                try:
                    api = HfApi()
                    api.upload_folder(
                        folder_path=INDEX_DIR,
                        repo_id=repo_id,
                        repo_type="dataset",
                        token=hf_token,
                        commit_message="Update FAISS index"
                    )
                    logger.info("Successfully uploaded index to Hugging Face Dataset.")
                except Exception as e:
                    logger.error(f"Failed to upload index to Hub: {e}", exc_info=True)
            else:
                logger.warning("DATASET_REPO_ID or HF_TOKEN not set. Skipping index upload.")
            # ---- END NEW CODE ----

        except Exception as e:
            logger.error(f"Error creating FAISS index: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error creating index: {str(e)}")

        logger.info(f"Successfully processed PDF: {len(chunks)} chunks, {len(chapters)} chapters")
        return {
            "status": "success", 
            "chunks": len(chunks),
            "chapters": chapters
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in process_pdf_endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        # Clean up temporary file
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.debug(f"Cleaned up temporary file: {file_path}")
            except Exception as e:
                logger.warning(f"Error removing temporary file {file_path}: {e}")

@app.post("/generate_question_paper/")
async def generate_question_paper(request_data: dict = Body(...)):
    """Generate question paper from processed PDF"""
    try:
        # Validate request data
        if not request_data or not isinstance(request_data, dict):
            logger.error("Invalid request_data: not a dictionary")
            raise HTTPException(status_code=400, detail="request_data must be a valid dictionary")
        
        if "numberOfPapers" not in request_data:
            logger.error("Missing 'numberOfPapers' in request_data")
            raise HTTPException(status_code=400, detail="Missing required field: numberOfPapers")
        
        num_papers = request_data.get("numberOfPapers")
        if not isinstance(num_papers, int) or num_papers <= 0:
            logger.error(f"Invalid numberOfPapers: {num_papers}")
            raise HTTPException(status_code=400, detail="numberOfPapers must be a positive integer")
        
        if "questions" not in request_data:
            logger.error("Missing 'questions' in request_data")
            raise HTTPException(status_code=400, detail="Missing required field: questions")
        
        if not isinstance(request_data["questions"], list) or len(request_data["questions"]) == 0:
            logger.error("Invalid questions: must be a non-empty list")
            raise HTTPException(status_code=400, detail="questions must be a non-empty list")
        
        logger.info(f"Generating {num_papers} question paper(s)")
        
        # Check if FAISS index exists
        if not os.path.exists(INDEX_DIR):
            logger.error(f"FAISS index not found: {INDEX_DIR}")
            raise HTTPException(
                status_code=404, 
                detail="No PDF has been processed yet. Please process a PDF first."
            )
        
        # Generate papers
        try:
            exam_paper = generate_multiple_papers_with_summaries(
                request_data, 
                num_papers, 
                k=10
            )
        except ValueError as e:
            logger.error(f"Invalid input for paper generation: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
        except FileNotFoundError as e:
            logger.error(f"FAISS index not found: {e}")
            raise HTTPException(
                status_code=404, 
                detail="No PDF has been processed yet. Please process a PDF first."
            )
        except Exception as e:
            logger.error(f"Error generating question papers: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error generating papers: {str(e)}")

        logger.debug(f"Generated exam_paper type: {type(exam_paper)}")
        logger.debug(f"Generated {len(exam_paper) if isinstance(exam_paper, list) else 0} papers")

        if not exam_paper or not isinstance(exam_paper, list):
            logger.warning("AI service did not return a valid array of papers")
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": "AI service did not return a valid array of papers."
                }
            )
        
        if len(exam_paper) == 0:
            logger.warning("No papers were generated")
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": "No papers were generated. Please check the request data and try again."
                }
            )

        logger.info(f"Successfully generated {len(exam_paper)} question paper(s)")
        return {
            "success": True,
            "question_paper": exam_paper
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_question_paper: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Internal server error: {str(e)}"
            }
        )
    
@app.post("/evaluate_answer_paper/")
async def evaluate_answer_paper(
    file : UploadFile = Form(...),
    question_paper_str: str = Form(...),
):
    """
    Endpoint to evaluate a student's handwritten answers.
    Expects:
    - question_paper: full JSON of generated exam paper
    - answer_paper: extracted text from student's handwritten answer sheet
    """
    try:
        # Save file locally
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        try:
            question_paper = json.loads(question_paper_str)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON format in question_paper: {e}")

        print("Evaluating Answer Paper...")
        extracted_answer_paper = extract_contents_from_pdf(file_path)
        if not extracted_answer_paper.strip():
            raise HTTPException(status_code=400, detail="Failed to extract text from the uploaded PDF file.")

        report = evaluate_answers(question_paper, extracted_answer_paper)
        logger.info(f"Evaluation successful for {file.filename}")
        print(report)

        # ✅ Optional: clean up temp file
        import os
        os.remove(file_path)
        return JSONResponse(content=report)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"CRITICAL ERROR in evaluate_answer_paper: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.get("/chunks/")
async def get_chunks(
    page: int = 1,
    page_size: int = 20,
    chapter_no: Optional[str] = None,
    subject: Optional[str] = None,
    className: Optional[str] = None
):
    """
    Retrieve chunks from the FAISS index with pagination and filtering.
    
    Query Parameters:
        page: Page number (default: 1)
        page_size: Number of chunks per page (default: 20, max: 100)
        chapter_no: Filter by chapter number (optional)
        subject: Filter by subject (optional)
        className: Filter by class name (optional)
    """
    try:
        # Validate inputs
        if page < 1:
            raise HTTPException(status_code=400, detail="page must be >= 1")
        
        if page_size < 1 or page_size > 100:
            raise HTTPException(status_code=400, detail="page_size must be between 1 and 100")
        
        # Check if FAISS index exists
        if not os.path.exists(INDEX_DIR):
            raise HTTPException(
                status_code=404,
                detail="No PDF has been processed yet. Please process a PDF first."
            )
        
        try:
            logger.info(f"Loading FAISS index from {INDEX_DIR}")
            vectorstore = FAISS.load_local(
                INDEX_DIR,
                embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error loading index: {str(e)}")
        
        # Get all documents from the vector store
        try:
            # FAISS doesn't have a direct "get all" method, so we use similarity search with a generic query
            # This is a workaround - in production, you might want to store chunks separately
            all_docs = vectorstore.similarity_search("", k=10000)  # Large k to get all docs
            
            # Filter by metadata if provided
            filtered_docs = []
            for doc in all_docs:
                metadata = doc.metadata
                
                # Apply filters
                if chapter_no and metadata.get("chapter_no") != chapter_no:
                    continue
                if subject and metadata.get("subject") != subject:
                    continue
                if className and metadata.get("class") != className:
                    continue
                
                filtered_docs.append(doc)
            
            # Calculate pagination
            total_chunks = len(filtered_docs)
            total_pages = (total_chunks + page_size - 1) // page_size
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            
            # Get paginated chunks
            paginated_docs = filtered_docs[start_idx:end_idx]
            
            # Format response
            chunks_data = []
            for doc in paginated_docs:
                chunks_data.append({
                    "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    "content_preview": doc.page_content[:200],
                    "metadata": {
                        "chapter_no": doc.metadata.get("chapter_no", ""),
                        "chapter_name": doc.metadata.get("chapter_name", ""),
                        "page": doc.metadata.get("page", 0),
                        "content_type": doc.metadata.get("content_type", ""),
                        "subject": doc.metadata.get("subject", ""),
                        "class": doc.metadata.get("class", ""),
                        "pdf_name": doc.metadata.get("pdf_name", ""),
                        "chunk_index": doc.metadata.get("chunk_index", None)
                    }
                })
            
            logger.info(f"Retrieved {len(paginated_docs)} chunks (page {page}/{total_pages})")
            
            return {
                "success": True,
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "total_chunks": total_chunks,
                    "total_pages": total_pages,
                    "has_next": page < total_pages,
                    "has_prev": page > 1
                },
                "filters": {
                    "chapter_no": chapter_no,
                    "subject": subject,
                    "className": className
                },
                "chunks": chunks_data
            }
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error retrieving chunks: {str(e)}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_chunks: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/chunks/stats/")
async def get_chunks_stats():
    """Get statistics about the chunks in the FAISS index"""
    try:
        if not os.path.exists(INDEX_DIR):
            raise HTTPException(
                status_code=404,
                detail="No PDF has been processed yet. Please process a PDF first."
            )
        
        try:
            vectorstore = FAISS.load_local(
                INDEX_DIR,
                embeddings,
                allow_dangerous_deserialization=True
            )
            all_docs = vectorstore.similarity_search("", k=10000)
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error loading index: {str(e)}")
        
        # Collect statistics
        total_chunks = len(all_docs)
        chapters = {}
        subjects = set()
        classes = set()
        content_types = {}
        
        for doc in all_docs:
            metadata = doc.metadata
            
            # Chapter stats
            chapter_key = f"{metadata.get('chapter_no', 'Unknown')} - {metadata.get('chapter_name', 'Unknown')}"
            chapters[chapter_key] = chapters.get(chapter_key, 0) + 1
            
            # Subject and class
            if metadata.get("subject"):
                subjects.add(metadata.get("subject"))
            if metadata.get("class"):
                classes.add(metadata.get("class"))
            
            # Content type stats
            content_type = metadata.get("content_type", "unknown")
            content_types[content_type] = content_types.get(content_type, 0) + 1
        
        return {
            "success": True,
            "stats": {
                "total_chunks": total_chunks,
                "total_chapters": len(chapters),
                "chapters": chapters,
                "subjects": list(subjects),
                "classes": list(classes),
                "content_types": content_types
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chunks stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


class DeleteRequest(BaseModel):
    pdf_name: str

@app.delete("/delete_book/")
async def delete_book_from_index(request: DeleteRequest):
    """
    Delete all chunks associated with a specific PDF from the FAISS index.
    """
    pdf_name = request.pdf_name
    
    if not os.path.exists(INDEX_DIR):
        raise HTTPException(status_code=404, detail="FAISS index not found")

    try:
        # 1. Load the index
        vectorstore = FAISS.load_local(
            INDEX_DIR, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # 2. Find IDs of chunks belonging to this PDF
        # LangChain's FAISS stores documents in .docstore._dict
        ids_to_delete = []
        for _id, doc in vectorstore.docstore._dict.items():
            if doc.metadata.get("pdf_name") == pdf_name:
                ids_to_delete.append(_id)
        
        if not ids_to_delete:
            return JSONResponse(content={"success": True, "message": "No chunks found for this book (already deleted?)"})

        # 3. Delete from Vector Store
        logger.info(f"Deleting {len(ids_to_delete)} chunks for {pdf_name}...")
        vectorstore.delete(ids_to_delete)
        
        # 4. Save updates to disk
        vectorstore.save_local(INDEX_DIR)
        
        # 5. (Optional) Re-upload to Hugging Face if configured
        repo_id = os.environ.get("DATASET_REPO_ID")
        hf_token = os.environ.get("HF_TOKEN")
        if repo_id and hf_token:
            try:
                api = HfApi()
                api.upload_folder(
                    folder_path=INDEX_DIR,
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=hf_token,
                    commit_message=f"Deleted book: {pdf_name}"
                )
            except Exception as e:
                logger.error(f"Failed to sync deletion to HF: {e}")

        return {"success": True, "message": f"Deleted {len(ids_to_delete)} chunks."}

    except Exception as e:
        logger.error(f"Error deleting book from index: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete from index: {str(e)}")

@app.delete("/clear_index/")
async def clear_index():
    """Wipes the vector store completely. Use this to reset."""
    if os.path.exists(INDEX_DIR):
        import shutil
        shutil.rmtree(INDEX_DIR)
        return {"message": "Index deleted. Now re-upload your PDF."}
    return {"message": "Index already empty."}

if __name__ == "__main__":
    import uvicorn
    # Note: Hugging Face Spaces will use the CMD in the Dockerfile (Gunicorn)
    # This block is for local development only.
    uvicorn.run("Final_code_Dev:app", host="0.0.0.0", port=8000, reload=True)