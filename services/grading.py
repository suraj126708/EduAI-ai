"""
Answer evaluation logic using Vertex AI.
"""
import json
import re
import time
import logging
import concurrent.futures
from typing import Dict, List
from pdf2image import convert_from_path
from google.genai import types
from dependencies import get_gemini_client
from services.pdf_processing import pil_to_part, safe_vertex_generate
from utils import clean_json_string, clean_latex_content

logger = logging.getLogger(__name__)

# Model name constant
MODEL_NAME = "gemini-3-flash-preview"


def extract_contents_from_pdf(file_path: str, max_workers: int = 5) -> str:
    """
    Extract all handwritten text from a student's handwritten answer sheet.
    Uses parallel processing to speed up extraction while maintaining quality.
    
    Args:
        file_path: Path to the PDF file
        max_workers: Maximum number of concurrent page processing workers (default: 5)
    
    Returns:
        Extracted text from all pages, sorted by page number
    """
    try:
        pages = convert_from_path(pdf_path=file_path) 
    except Exception as e:
        logger.error(f"Error converting PDF to images: {e}")
        raise RuntimeError(f"Could not convert PDF. Is Poppler installed? Error: {e}")

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

    def process_single_page(page_data):
        """Process a single page and return (page_num, content) tuple"""
        page_num, page = page_data
        try:
            response = safe_vertex_generate([prompt, pil_to_part(page)])
            
            # Handle response - check both response.text and response.candidates for compatibility
            content = None
            if response:
                if hasattr(response, 'text') and response.text:
                    content = response.text.strip()
                elif hasattr(response, 'candidates') and response.candidates:
                    if (len(response.candidates) > 0 and 
                        hasattr(response.candidates[0], 'content') and
                        response.candidates[0].content.parts):
                        content = response.candidates[0].content.parts[0].text.strip()
            
            if content:
                if content.startswith("```"):
                    content = content.strip("`").replace("json", "").strip()
                return (page_num, content)
            else:
                logger.warning(f"Empty response for page {page_num}")
                return (page_num, "")
        except Exception as e:
            logger.error(f"Error processing page {page_num}: {e}")
            return (page_num, "")

    # Process pages in parallel
    logger.info(f"Processing {len(pages)} pages in parallel (max {max_workers} workers)...")
    page_data_list = [(page_num, page) for page_num, page in enumerate(pages, start=1)]
    
    # Use ThreadPoolExecutor for parallel processing
    page_results = []
    effective_workers = min(max_workers, len(pages))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = {executor.submit(process_single_page, page_data): page_data[0] 
                   for page_data in page_data_list}
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result:
                    page_results.append(result)
            except Exception as e:
                page_num = futures.get(future, "unknown")
                logger.error(f"Error in future result for page {page_num}: {e}", exc_info=True)
    
    # Sort results by page number to maintain document order
    page_results.sort(key=lambda x: x[0])
    
    # Combine all pages in order
    extracted_text = ""
    for page_num, content in page_results:
        if content:
            extracted_text += f"\n\n--- Page {page_num} ---\n{content}"
    
    logger.info(f"Extracted {len(extracted_text)} characters from answer sheet ({len(pages)} pages).")
    return extracted_text


def assign_marks(question: str, correct_answer: str, student_answer: str, max_marks: int) -> Dict:
    """Uses Vertex AI (Gemini) to evaluate the student's handwritten answer and assign marks."""
    prompt = f"""
You are a strict, rule-bound examiner evaluating a student's handwritten answer sheet.

Question: {question}
Correct Answer: {correct_answer}
Student's Handwritten Answer: {student_answer}

---

### 🔥 OVERALL EVALUATION PRINCIPLE:
Award marks **only for what is explicitly written by the student**, not for what they "might have meant".  
No assumptions. No generosity. No filling gaps.

---

## 🎯 MARKING RULES (STRICTER VERSION)

Note : if the student answer is Empty or Blank ( "" ) , award 0 marks. [ VERY VERY IMPORTANT ]

### 1. Zero-tolerance for missing required points  
- If the question demands **specific items** (e.g., "herders, farmers, merchants, kings"), the answer MUST explicitly mention them.  
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
- Vague statements that don't show clear understanding → **0**.

### 5. Specificity Required  
- General statements like "people lived differently" or "past was different for everyone" are NOT enough for 3+ mark questions.
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
        # Build content from prompt
        content = [types.Part(text=prompt)]
        response = safe_vertex_generate(content)
        
        if not response:
            logger.warning("No response from Gemini for mark assignment")
            return {"awarded_marks": 0, "remarks": "Error: No response from AI"}
        
        # Handle both response formats (response.text and response.candidates)
        raw = None
        if hasattr(response, 'text') and response.text:
            raw = response.text
        elif hasattr(response, 'candidates') and response.candidates:
            if (len(response.candidates) > 0 and 
                hasattr(response.candidates[0], 'content') and
                response.candidates[0].content.parts):
                raw = response.candidates[0].content.parts[0].text
        
        if not raw:
            logger.warning("No text in Gemini response")
            return {"awarded_marks": 0, "remarks": "Error: No text in response"}
        if not raw:
            logger.warning("Empty text in Vertex AI response")
            return {"awarded_marks": 0, "remarks": "Error: Empty text in response"}
        
        raw = raw.strip()
        logger.info(f"Raw Evaluation Output (first 500 chars): {raw[:500]}")

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "awarded_marks" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass
        
        cleaned = raw
        
        if "```json" in cleaned:
            cleaned = re.sub(r'```json\s*', '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'```\s*$', '', cleaned, flags=re.MULTILINE)
        elif "```" in cleaned:
            cleaned = re.sub(r'```\s*', '', cleaned)
        
        cleaned = cleaned.strip()
        
        json_match = re.search(r'\{[\s\S]*\}', cleaned)
        if json_match:
            cleaned = json_match.group(0)
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict) and "awarded_marks" in parsed:
                    return parsed
            except json.JSONDecodeError as e:
                logger.debug(f"JSON decode error after regex extraction: {e}")
                try:
                    cleaned_json = clean_json_string(cleaned)
                    parsed = json.loads(cleaned_json)
                    if isinstance(parsed, dict) and "awarded_marks" in parsed:
                        return parsed
                except Exception as e2:
                    logger.debug(f"clean_json_string also failed: {e2}")
        
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
        
        logger.error(f"Failed to parse JSON from response. Full response (first 1000 chars): {raw[:1000]}")
        return {"awarded_marks": 0, "remarks": f"Error: Could not parse JSON from AI response. Response length: {len(raw)} chars"}
            
    except Exception as e:
        logger.error(f"Error in assign_marks: {e}", exc_info=True)
        return {"awarded_marks": 0, "remarks": f"Error: {str(e)}"}


def retrieve_section_answers(section_title: str, questions: List[Dict], answer_paper: str) -> Dict:
    """Uses Vertex AI (Gemini) once per section to extract answers for all questions."""
    q_descriptions = []
    for q in questions:
        q_snippet = q['question'][:50] + "..." if len(q['question']) > 50 else q['question']
        q_descriptions.append(f"Q{q['questionNo']}: {q_snippet}")
    
    questions_summary = "\n".join(q_descriptions)

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
        # Build content from prompt
        content = [types.Part(text=prompt)]
        response = safe_vertex_generate(content)
        
        # Handle both response formats (response.text and response.candidates)
        text = None
        if response:
            if hasattr(response, 'text') and response.text:
                text = response.text.strip()
            elif hasattr(response, 'candidates') and response.candidates:
                if (len(response.candidates) > 0 and 
                    hasattr(response.candidates[0], 'content') and
                    response.candidates[0].content.parts):
                    text = response.candidates[0].content.parts[0].text.strip()
        
        if not text:
            logger.warning("Empty response from Gemini for answer extraction")
            return {}

        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            text = match.group(0)

        text = clean_json_string(text)
        
        data = json.loads(text)
        cleaned_answers = {}
        for a in data.get("answers", []):
            raw_ans = a.get("studentAnswer", "")
            cleaned_answers[a["questionNo"]] = clean_latex_content(raw_ans)
            
        return cleaned_answers

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON returned by Vertex AI. Full response:\n{text if 'text' in locals() else 'No response'}")
        return {}
    except Exception as e:
        logger.error(f"Error retrieving section answers: {e}", exc_info=True)
        return {}


def analyze_chapters_with_llm(report: Dict) -> Dict:
    """Call LLM (Vertex AI) to compute chapter-wise totals and produce strengths/weaknesses/recommendations."""
    per_question_list = []
    for section in report.get("sections", []):
        for q in section.get("questions", []):
            per_question_list.append({
                "questionNo": q.get("questionNo"),
                "chapterNo": q.get("chapterNo", q.get("chapter", "Unknown")),
                "marks": q.get("marks", 0),
                "awarded": q.get("awarded", 0),
                "studentAnswer": q.get("studentAnswer", "")
            })

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
        # Build content from prompt
        content = [types.Part(text=prompt)]
        response = safe_vertex_generate(content)
        
        # Handle both response formats (response.text and response.candidates)
        text = None
        if response:
            if hasattr(response, 'text') and response.text:
                text = response.text.strip()
            elif hasattr(response, 'candidates') and response.candidates:
                if (len(response.candidates) > 0 and 
                    hasattr(response.candidates[0], 'content') and
                    response.candidates[0].content.parts):
                    text = response.candidates[0].content.parts[0].text.strip()
        
        if not text:
            logger.warning("No response from Gemini for chapter analysis")
            return {}
        cleaned_text = clean_json_string(text)
        return json.loads(cleaned_text)

    except Exception as e:
        logger.error(f"Chapter analysis failed: {e}", exc_info=True)
        return {
            "chapters": [],
            "overall_summary": {
                "strong_chapters": [],
                "weak_chapters": [],
                "study_plan": []
            }
        }


def evaluate_answers(question_paper: Dict, answer_paper: str, max_workers: int = 10) -> Dict:
    """
    Evaluate student's handwritten answers with the official question paper.
    Uses parallel processing to evaluate multiple questions concurrently.
    
    Args:
        question_paper: Dictionary containing the question paper structure
        answer_paper: Extracted text from student's handwritten answer sheet (handles answers across different pages)
        max_workers: Maximum number of parallel workers for evaluation (default: 10)
    
    Returns:
        Dictionary containing evaluation results with chapter-wise analysis
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
            raw_ans = q.get('correct answer') or q.get('correct_answer') or q.get('correctAnswer')
            correct_ans = clean_latex_content(raw_ans)
            clean_student_ans = clean_latex_content(student_ans)
            
            evaluation = assign_marks(q_text, correct_ans, clean_student_ans, marks)
            
            logger.debug(f"Evaluation for Q{q['questionNo']}: {evaluation}")

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
                "chapterNo": q.get('chapterNo', q.get('chapter', 'Unknown')),
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
                "chapterNo": q.get('chapterNo', 'Unknown'),
                "studentAnswer": student_ans,
                "correctAnswer": q.get('correct_answer', ''),
                "awarded": 0,
                "remarks": f"Error during evaluation: {str(e)}"
            }

    for section in question_paper['sections']:
        sec_title = section.get('sectionTitle') or section.get('sectionName') or section.get('title') or "Untitled Section"
        
        section_result = {"sectionTitle": sec_title, "questions": []}

        answers_map = retrieve_section_answers(
            sec_title, section.get('questions', []), answer_paper
        )

        logger.info(f"Answers Map for section '{sec_title}': {answers_map}")

        questions_to_evaluate = []
        for q in section.get('questions',[]):
            student_ans = answers_map.get(q['questionNo'], "")
            questions_to_evaluate.append((q, student_ans))
            total_marks += q.get('marks', 0)

        if questions_to_evaluate:
            try:
                effective_workers = min(max_workers, len(questions_to_evaluate))
                with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as executor:
                    futures = {
                        executor.submit(evaluate_single_question, q, student_ans): q
                        for q, student_ans in questions_to_evaluate
                    }
                    
                    evaluated_questions = []
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            question_result = future.result()
                            evaluated_questions.append(question_result)
                            obtained_marks += question_result.get("awarded", 0)
                        except Exception as e:
                            q = futures.get(future, {})
                            logger.error(f"Error in future result for question {q.get('questionNo', 'unknown')}: {e}", exc_info=True)
                    
                    evaluated_questions.sort(key=lambda x: x.get("questionNo", ""))
                    section_result['questions'] = evaluated_questions
                    
            except Exception as e:
                logger.error(f"Error in parallel evaluation for section '{section['sectionTitle']}': {e}", exc_info=True)
                for q, student_ans in questions_to_evaluate:
                    question_result = evaluate_single_question(q, student_ans)
                    section_result['questions'].append(question_result)
                    obtained_marks += question_result.get("awarded", 0)

        result.append(section_result)

    final_report = {
        "Subject": question_paper["subject"],
        "Class": question_paper["className"],
        "totalMarks": total_marks,
        "obtainedMarks": int(obtained_marks),
        "sections": result
    }

    logger.info("Running chapter-wise analysis...")
    chapter_summary = analyze_chapters_with_llm(final_report)
    final_report["chapter_summary"] = chapter_summary

    logger.info(f"Evaluation complete: {obtained_marks}/{total_marks} marks")
    return final_report

