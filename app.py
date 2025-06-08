import streamlit as st
import torch
import os
import sys
import logging
from datetime import datetime
import asyncio
import nest_asyncio
import platform
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Handle Windows event loop policy
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def check_environment():
    """Check if all required components are available"""
    st.sidebar.subheader("🔍 Environment Check")
    
    # Check Python version
    st.sidebar.write(f"Python: {sys.version[:5]}")
    
    # Check PyTorch
    try:
        st.sidebar.write(f"PyTorch: {torch.__version__}")
        st.sidebar.write(f"CUDA: {'✅' if torch.cuda.is_available() else '❌ (CPU only)'}")
    except Exception as e:
        st.sidebar.error(f"PyTorch issue: {e}")
        return False
    
    # Check transformers
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        st.sidebar.write("Transformers: ✅")
    except Exception as e:
        st.sidebar.error(f"Transformers issue: {e}")
        return False
    
    # Check helper functions
    try:
        from src.helper import preprocess_query, clean_sql_output, format_t5_prompt
        st.sidebar.write("Helper functions: ✅")
    except Exception as e:
        st.sidebar.error(f"Helper functions issue: {e}")
        st.error("⚠️ Cannot import helper functions. Please ensure src/helper.py exists with required functions:")
        st.code("""
# Required functions in src/helper.py:
def preprocess_query(query):
    # Your preprocessing logic
    return query

def clean_sql_output(sql):
    # Your SQL cleaning logic
    return sql

def format_t5_prompt(query):
    # Your T5 prompt formatting logic
    return f"translate English to SQL: {query}"
        """)
        return False
    
    # Check model directory
    model_path = "final_model"
    if os.path.exists(model_path):
        st.sidebar.write("Model directory: ✅")
        model_files = os.listdir(model_path)
        st.sidebar.write(f"Model files: {len(model_files)} files")
    else:
        st.sidebar.error("❌ final_model directory not found")
        return False
    
    return True

class SQLConverter:
    def __init__(self):
        self.model_name = "t5-large"
        self.max_length = 256
        # Force CPU for Streamlit Cloud to avoid CUDA issues
        self.device = torch.device("cpu")
        self.tokenizer = None
        self.model = None
        
        try:
            # Import helper functions
            from src.helper import preprocess_query, clean_sql_output, format_t5_prompt
            self.preprocess_query = preprocess_query
            self.clean_sql_output = clean_sql_output
            self.format_t5_prompt = format_t5_prompt
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model
            logger.info("Loading model...")
            from transformers import AutoModelForSeq2SeqLM
            
            model_path = "final_model"
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                local_files_only=True  # Ensure we use local files
            )
            
            self.model.to(self.device)
            self.model.eval()
            
            # Clean up memory
            gc.collect()
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error initializing SQLConverter: {str(e)}")
            raise e

    def convert_to_sql(self, natural_query: str) -> str:
        try:
            if not self.model or not self.tokenizer:
                return "Error: Model not properly loaded"
            
            processed_query = self.preprocess_query(natural_query)
            input_text = self.format_t5_prompt(processed_query)
            
            inputs = self.tokenizer(
                input_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.max_length,
                    num_beams=4,  # Reduced for memory efficiency
                    temperature=0.3,  
                    do_sample=False, 
                    early_stopping=True,
                    repetition_penalty=1.1
                )

            sql_query = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up memory
            del inputs, input_ids, attention_mask, outputs
            gc.collect()
            
            return self.clean_sql_output(sql_query)
            
        except Exception as e:
            logger.error(f"Error in convert_to_sql: {str(e)}")
            return f"Error converting query: {str(e)}"

def main():
    st.set_page_config(
        page_title="Natural Language to SQL Converter", 
        page_icon="🔄",
        layout="wide"
    )
    
    # Check environment first
    env_ok = check_environment()
    
    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'converter_loaded' not in st.session_state:
        st.session_state.converter_loaded = False
    
    # History sidebar
    with st.sidebar:
        st.subheader("📋 Query History")
        if not st.session_state.history:
            st.info("No queries yet")
        else:
            # Show last 5 queries
            recent_history = st.session_state.history[-5:]
            for idx, (timestamp, query, sql) in enumerate(reversed(recent_history)):
                with st.expander(f"Query {len(recent_history) - idx}"):
                    st.text(f"Time: {timestamp}")
                    st.write("**Natural Query:**")
                    st.info(query)
                    st.write("**SQL Query:**")
                    st.code(sql, language="sql")
        
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
    
    # Main interface
    st.title("🔄 Natural Language to SQL Converter")
    st.write("Convert your natural language questions into SQL queries using AI!")
    
    if not env_ok:
        st.error("❌ Environment check failed. Please check the sidebar for details.")
        return
    
    # Load converter
    @st.cache_resource
    def load_converter():
        try:
            with st.spinner("🔄 Loading AI model... This may take a moment."):
                converter = SQLConverter()
                st.success("✅ Model loaded successfully!")
                return converter
        except Exception as e:
            st.error(f"❌ Failed to load model: {str(e)}")
            logger.error(f"Model loading failed: {str(e)}")
            return None
    
    converter = load_converter()
    
    if converter is None:
        st.error("Could not load the model. This might be due to:")
        st.write("- Missing model files")
        st.write("- Memory limitations")
        st.write("- Missing dependencies")
        st.info("Please check the sidebar for detailed error information.")
        return
    
    # Input section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        natural_query = st.text_area(
            "Enter your natural language question:",
            height=100,
            placeholder="Example: Show me all customers who ordered more than 5 items last month"
        )
    
    with col2:
        st.write("**Example queries:**")
        st.write("• Show all users")
        st.write("• Find top 10 products")
        st.write("• Count orders by month")
        st.write("• List customers from NYC")
    
    # Convert button
    if st.button("🚀 Convert to SQL", type="primary", use_container_width=True):
        if natural_query.strip():
            with st.spinner("🧠 Converting your query..."):
                try:
                    sql_query = converter.convert_to_sql(natural_query)
                    
                    # Display result
                    st.subheader("📝 Generated SQL Query:")
                    st.code(sql_query, language="sql")
                    
                    # Add to history
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.history.append((timestamp, natural_query, sql_query))
                    
                    # Limit history size to prevent memory issues
                    if len(st.session_state.history) > 20:
                        st.session_state.history = st.session_state.history[-20:]
                    
                    st.success("✅ Query converted successfully!")
                    
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
                    logger.error(f"Conversion error: {str(e)}")
        else:
            st.warning("⚠️ Please enter a question first!")
    
    # Footer
    st.markdown("---")
    st.markdown("💡 **Tip:** Be specific in your queries for better SQL generation!")

if __name__ == "__main__":
    try:
        # Handle async event loop for Streamlit
        try:
            asyncio.get_running_loop()
            nest_asyncio.apply()
        except RuntimeError:
            pass
        
        main()
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Main application error: {str(e)}")