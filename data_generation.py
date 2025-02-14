import csv
import random
from faker import Faker
from datetime import datetime

# Initialize Faker
fake = Faker()

# Define possible values
departments = ['Engineering', 'Marketing', 'Sales', 'HR', 'Finance', 'IT Support']
positions = ['Manager', 'Engineer', 'Specialist', 'Director', 'Coordinator']
conditions = ['greater than', 'less than', 'equal to', 'not equal to', 'after', 'before']
fields = ['name', 'age', 'email', 'salary', 'department', 'position', 'hire_date']

def generate_dataset(num_samples=10000, filename='nl_sql_dataset.csv'):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id', 'natural_language', 'sql_query']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in range(1, num_samples + 1):
            # Randomly select query components
            num_fields = random.randint(1, 4)
            selected_fields = random.sample(fields, num_fields)
            condition_field = random.choice(fields)
            condition_operator = random.choice(conditions)
            condition_value = ''

            # Generate condition value based on field type
            if condition_field == 'department':
                condition_value = random.choice(departments)
            elif condition_field == 'position':
                condition_value = random.choice(positions)
            elif condition_field == 'age':
                condition_value = random.randint(22, 65)
            elif condition_field == 'salary':
                condition_value = random.randint(40000, 150000)
            elif condition_field == 'hire_date':
                condition_value = fake.date_between(start_date='-10y', end_date='today').strftime('%Y-%m-%d')
            else:
                condition_value = fake.name()
            
            # Construct natural language query
            nl_query = f"Show me the {', '.join(selected_fields)} of employees where {condition_field} is {condition_operator} {condition_value}"
            
            # Map operator to SQL syntax
            operator_mapping = {
                'greater than': '>',
                'less than': '<',
                'equal to': '=',
                'not equal to': '!=',
                'after': '>',
                'before': '<'
            }
            sql_operator = operator_mapping.get(condition_operator, '=')
            
            # Handle dates without quotes
            if condition_field == 'age' or condition_field == 'salary':
                sql_condition = f"{condition_field} {sql_operator} {condition_value}"
            else:
                sql_condition = f"{condition_field} {sql_operator} '{condition_value}'"
            
            # Construct SQL query
            select_clause = ', '.join(selected_fields)
            sql_query = f"SELECT {select_clause} FROM users WHERE {sql_condition};"
            
            # Write to CSV
            writer.writerow({
                'id': i,
                'natural_language': nl_query,
                'sql_query': sql_query
            })
    print(f"Dataset of {num_samples} samples created as {filename}")

if __name__ == "__main__":
    generate_dataset()
