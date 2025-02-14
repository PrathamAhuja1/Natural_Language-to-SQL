import os
from dotenv import load_dotenv
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

from src.helper import (
    validate_sql,
    preprocess_query,
    clean_sql_output,
    log_training_metrics,
    timestamp,
    get_logger
)

# Get logger instance
logger = get_logger(__name__)

class SQLModelTrainer:
    """Handles model setup and training for SQL generation"""
    
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Configuration
        self.model_name = "codellama/CodeLlama-7b-Instruct-hf"
        self.output_dir = f"sql_trained_model_{timestamp()}"
        self.max_length = 512

        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components
        self._setup_model_components()

    def _setup_model_components(self):
        """Configures model with proper quantization"""
        try:
            # 1. Configure 4-bit quantization
            self.quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            
            # 2. Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.hf_token,
                padding_side="left",
                truncation_side="left"
            )
            
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 3. Load model with quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=self.hf_token,
                quantization_config=self.quant_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            # 4. Prepare for PEFT training
            self.model = prepare_model_for_kbit_training(self.model)
            
            # 5. Configure LoRA
            self.lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            self.model = get_peft_model(self.model, self.lora_config)
            
            # Log trainable parameters
            self.model.print_trainable_parameters()
            logger.info("Model setup successful")

        except Exception as e:
            logger.error(f"Model setup failed: {str(e)}")
            raise

    def prepare_training_data(self, data_path: str):
        """Processes and formats training data"""
        try:
            # Load dataset
            dataset = load_dataset('csv', data_files={'train': data_path})
            dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)
            
            # Training template
            prompt_template = (
                "[INST] <<SYS>>\n"
                "Generate SQL query for this request\n"
                "<</SYS>>\n\n"
                "{natural_language}\n[/INST]\n"
                "{sql_query}"
            )

            def format_example(examples):
                """Formats individual training examples"""
                processed_inputs = []
                
                for nl, sql in zip(examples['natural_language'], examples['sql_query']):
                    # Validate SQL structure
                    is_valid, msg = validate_sql(sql)
                    if not is_valid:
                        logger.warning(f"Skipping invalid SQL: {msg}")
                        continue
                    
                    # Process and format the example
                    processed_inputs.append(
                        prompt_template.format(
                            natural_language=preprocess_query(nl),
                            sql_query=clean_sql_output(sql)
                        )
                    )
                
                # Tokenize inputs
                tokenized = self.tokenizer(
                    processed_inputs,
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors=None
                )
                
                # Create labels
                tokenized["labels"] = tokenized["input_ids"].copy()
                return tokenized

            # Process dataset
            processed_dataset = dataset.map(
                format_example,
                batched=True,
                remove_columns=dataset['train'].column_names,
                num_proc=4
            )
            
            # Log processing metrics
            log_training_metrics({
                "total_examples": len(processed_dataset['train']),
                "validation_examples": len(processed_dataset['test'])
            }, prefix="data_processing")
            
            return processed_dataset
            
        except Exception as e:
            logger.error(f"Data processing failed: {str(e)}")
            raise

    def execute_training(self, train_data, val_data):
        """Manages the training process"""
        try:
            # Create logging directory
            log_dir = os.path.join(self.output_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
            
            # Training configuration
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                num_train_epochs=10,
                learning_rate=1e-4,
                gradient_accumulation_steps=4,
                fp16=True,
                optim="paged_adamw_8bit",
                logging_steps=10,
                evaluation_strategy="steps",
                eval_steps=50,
                save_strategy="steps",
                save_steps=50,
                load_best_model_at_end=True,
                logging_dir=log_dir,
                report_to="tensorboard",
                remove_unused_columns=False
            )

            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_data,
                eval_dataset=val_data,
                data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
            )

            # Start training
            logger.info(f"Training started at {timestamp()}")
            trainer.train()
            
            # Save model
            save_path = os.path.join(self.output_dir, "final_model")
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            
            logger.info(f"Training complete! Model saved to {save_path}")

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

def main():
    """Main execution flow"""
    try:
        # Initialize trainer
        trainer = SQLModelTrainer()
        
        # Prepare data
        dataset = trainer.prepare_training_data("nl_sql_dataset.csv")
        
        # Execute training
        trainer.execute_training(dataset['train'], dataset['test'])
        
    except Exception as e:
        logger.error(f"Program error: {str(e)}")
        raise

if __name__ == "__main__":
    main()