import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
from src.helper import preprocess_query, clean_sql_output, format_t5_prompt
from datetime import datetime

class SQLConverter:
    def __init__(self):
        self.model_name = "t5-large"
        self.max_length = 256
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained("final_model")
        self.model.to(self.device)
        self.model.eval()

    def convert_to_sql(self, natural_query: str) -> str:
        try:
            # Preprocess the query and format it for T5
            processed_query = preprocess_query(natural_query)
            input_text = format_t5_prompt(processed_query)
            
            # Tokenize input
            inputs = self.tokenizer(
                input_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Move input tensors to the same device as model
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            # Generate SQL query
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.max_length,
                    num_beams=5,
                    early_stopping=True
                )
            
            # Decode the generated SQL
            sql_query = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return clean_sql_output(sql_query)
            
        except Exception as e:
            return f"Error converting query: {str(e)}"

def main():
    st.set_page_config(page_title="Natural Language to SQL Converter", page_icon="🔄")
    
    # Initialize session state for history if it doesn't exist
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Sidebar for history
    with st.sidebar:
        st.title("Query History")
        if not st.session_state.history:
            st.info("No queries yet")
        else:
            for idx, (timestamp, query, sql) in enumerate(reversed(st.session_state.history)):
                with st.expander(f"Query {len(st.session_state.history) - idx}"):
                    st.text(f"Time: {timestamp}")
                    st.write("Natural Query:")
                    st.info(query)
                    st.write("SQL Query:")
                    st.code(sql, language="sql")
    
    # Main content
    st.title("Natural Language to SQL Converter")
    st.write("Convert your natural language questions into SQL queries!")
    
    # Initialize the converter
    @st.cache_resource
    def load_converter():
        return SQLConverter()
    
    converter = load_converter()
    
    # Input text area for natural language query
    natural_query = st.text_area(
        "Enter your question:",
        placeholder="Example: Show me all customers from New York who made purchases last month",
        height=100
    )
    
    # Convert button
    if st.button("Convert to SQL"):
        if natural_query:
            with st.spinner("Converting your query..."):
                try:
                    sql_query = converter.convert_to_sql(natural_query)
                    
                    # Display the result
                    st.subheader("Generated SQL Query:")
                    st.code(sql_query, language="sql")
                    
                    # Add a copy button
                    st.button(
                        "Copy SQL",
                        on_click=lambda: st.write(
                            f'<script>navigator.clipboard.writeText("{sql_query}")</script>',
                            unsafe_allow_html=True
                        )
                    )
                    
                    # Add to history
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.history.append((timestamp, natural_query, sql_query))
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a question first!")

if __name__ == "__main__":
    main()