import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import PyPDF2
import os
import tempfile
import logging
from typing import Optional, Dict, Any
import gc
import torch

# ===== CONFIGURATION =====
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# App configuration
st.set_page_config(
    page_title="Smart Resume Assistant", 
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "Smart Resume Assistant - Enhance and parse resumes with AI"
    }
)

# Constants
MODEL_PATH = "Resume_Enhancer/Model"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
SUPPORTED_EXTENSIONS = ["pdf"]
MAX_ENHANCEMENT_LENGTH = 150
MIN_INPUT_LENGTH = 5

# ===== UTILITY FUNCTIONS =====
def validate_file(uploaded_file) -> tuple[bool, str]:
    """Validate uploaded file size and type."""
    if uploaded_file.size > MAX_FILE_SIZE:
        return False, f"File size too large. Maximum allowed: {MAX_FILE_SIZE // (1024*1024)}MB"
    
    if uploaded_file.type != "application/pdf":
        return False, "Only PDF files are supported"
    
    return True, ""

def extract_text_from_pdf(file_path: str) -> tuple[str, bool]:
    """Extract text from PDF with error handling."""
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            
            # Check if PDF is encrypted
            if reader.is_encrypted:
                return "", False
            
            text = ""
            for page_num, page in enumerate(reader.pages):
                try:
                    text += page.extract_text() + "\n"
                except Exception as e:
                    logger.warning(f"Error extracting text from page {page_num}: {e}")
                    continue
            
            return text.strip(), True
    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
        return "", False

def clean_input_text(text: str) -> str:
    """Clean and validate input text."""
    return text.strip()

# ===== CACHED MODEL LOADING =====
@st.cache_resource(show_spinner="Loading AI model...")
def load_enhancer_model():
    """Load the resume enhancement model with error handling."""
    try:
        # Check if model path exists
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model not found at {MODEL_PATH}. Please check the model path.")
            return None
        
        # Load with optimizations
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        pipe = pipeline(
            "text2text-generation", 
            model=model, 
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        logger.info("Model loaded successfully")
        return pipe
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        st.error(f"Failed to load model: {e}")
        return None

@st.cache_data(show_spinner="Parsing resume...")
def parse_resume_cached(text_data: str) -> Dict[Any, Any]:
    """Cached resume parsing function."""
    try:
        # Import here to avoid loading issues if module doesn't exist
        from Resume_LLM.uk_resume_data_extraction import extract_uk_resume_data
        return extract_uk_resume_data(text_data)
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return {"error": "Resume parsing module not found"}
    except Exception as e:
        logger.error(f"Parsing error: {e}")
        return {"error": f"Failed to parse resume: {str(e)}"}

# ===== MAIN APP =====
def main():
    st.title("🧠 Smart Resume Assistant")
    st.markdown("---")
    
    # Mode selection with better UI
    mode = st.radio(
        "Choose Mode",
        ["Resume Enhancer", "Resume Parser"],
        help="Select between enhancing resume text or parsing uploaded resumes"
    )
    
    if mode == "Resume Enhancer":
        render_enhancer_mode()
    else:
        render_parser_mode()

def render_enhancer_mode():
    """Render the resume enhancer interface."""
    st.subheader("✨ Resume Line Enhancer")
    st.markdown("Transform basic resume bullets into professional, impactful statements.")
    
    # Input section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        text_input = st.text_area(
            "Enter resume content to enhance:",
            placeholder="e.g., 'worked on data analysis projects'",
            height=100,
            help="Enter a resume bullet point or description to make it more professional"
        )
    
    with col2:
        st.markdown("### Tips")
        st.info(
            "✅ Be specific\n\n"
            "✅ Include metrics\n\n" 
            "✅ Use action verbs\n\n"
            "✅ Keep it concise"
        )
    
    # Enhancement controls
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        enhance_button = st.button("🚀 Enhance", type="primary")
    
    with col2:
        max_length = st.slider("Max Length", 50, 200, MAX_ENHANCEMENT_LENGTH, step=10)
    
    # Process enhancement
    if enhance_button:
        if not text_input or len(text_input.strip()) < MIN_INPUT_LENGTH:
            st.warning(f"Please enter at least {MIN_INPUT_LENGTH} characters.")
            return
        
        clean_input = clean_input_text(text_input)
        
        # Load model
        enhancer = load_enhancer_model()
        if enhancer is None:
            return
        
        try:
            with st.spinner("Enhancing your resume content..."):
                result = enhancer(
                    clean_input,
                    max_length=max_length,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7
                )[0]["generated_text"]
            
            # Display results
            st.success("✅ Enhanced Version:")
            st.markdown(f"**Original:** {clean_input}")
            st.markdown(f"**Enhanced:** {result}")
            
            # Copy button (requires streamlit-extras or custom JS)
            st.code(result, language=None)
            
        except Exception as e:
            st.error(f"Enhancement failed: {str(e)}")
            logger.error(f"Enhancement error: {e}")
        finally:
            # Clean up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

def render_parser_mode():
    """Render the resume parser interface."""
    st.subheader("📄 Resume Parser")
    st.markdown("Upload a PDF resume to extract and structure key information.")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a PDF resume file",
        type=SUPPORTED_EXTENSIONS,
        help=f"Maximum file size: {MAX_FILE_SIZE // (1024*1024)}MB"
    )
    
    if uploaded_file is not None:
        # Validate file
        is_valid, error_msg = validate_file(uploaded_file)
        if not is_valid:
            st.error(error_msg)
            return
        
        # Display file info
        st.info(f"📎 File: {uploaded_file.name} ({uploaded_file.size // 1024} KB)")
        
        # Extract text
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        try:
            with st.spinner("Extracting text from PDF..."):
                text_data, success = extract_text_from_pdf(tmp_file_path)
            
            if not success:
                st.error("Failed to extract text from PDF. The file might be corrupted or encrypted.")
                return
            
            if not text_data:
                st.warning("No text found in the PDF. The file might be image-based or empty.")
                return
            
            # Display extracted text
            with st.expander("📝 Extracted Text", expanded=False):
                st.text_area("", text_data, height=200, disabled=True)
            
            # Parse button
            if st.button("🔍 Parse Resume", type="primary"):
                with st.spinner("Analyzing resume structure..."):
                    parsed_data = parse_resume_cached(text_data)
                
                if "error" in parsed_data:
                    st.error(parsed_data["error"])
                else:
                    st.success("✅ Resume Successfully Parsed!")
                    
                    # Display parsed data in a nice format
                    if parsed_data:
                        for key, value in parsed_data.items():
                            if value:  # Only show non-empty fields
                                st.markdown(f"**{key.title().replace('_', ' ')}:**")
                                if isinstance(value, list):
                                    for item in value:
                                        st.markdown(f"• {item}")
                                else:
                                    st.markdown(f"{value}")
                                st.markdown("---")
                    else:
                        st.info("No structured data could be extracted from the resume.")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"Parser error: {e}")
        
        finally:
            # Cleanup temporary file
            try:
                os.unlink(tmp_file_path)
            except:
                pass

# ===== SIDEBAR =====
def render_sidebar():
    """Render sidebar with additional info."""
    with st.sidebar:
        st.markdown("## ℹ️ About")
        st.markdown(
            "This app helps you enhance resume content and parse resume structure using AI."
        )
        
        st.markdown("## 🔧 Features")
        st.markdown(
            "- **Resume Enhancer**: Transform basic text into professional resume content\n"
            "- **Resume Parser**: Extract structured data from PDF resumes"
        )
        
        st.markdown("## 💡 Tips")
        st.markdown(
            "- Use specific, measurable achievements\n"
            "- Include relevant keywords for your industry\n"
            "- Keep descriptions concise but impactful"
        )

# ===== RUN APP =====
if __name__ == "__main__":
    render_sidebar()
    main()