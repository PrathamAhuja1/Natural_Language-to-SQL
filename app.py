from src.converter import NLToSQLConverter
import logging

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    try:
        converter = NLToSQLConverter(
            model_path="results/final_model",
            db_path="data/users.db"
        )

        while True:
            query = input("\nEnter your query (or 'exit' to quit): ")
            if query.lower() == 'exit':
                break

            result = converter.process_query(query)
            print(f"\nGenerated SQL: {result.sql_query}")
            
            if result.error:
                print(f"Error: {result.error}")
            else:
                print("\nResults:")
                for row in result.results:
                    print(row)

    except Exception as e:
        logger.error(f"Application error: {e}")

if __name__ == "__main__":
    main()