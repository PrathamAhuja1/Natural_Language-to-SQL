import csv
import random
from faker import Faker
from datetime import datetime, timedelta
import itertools

# Initialize Faker with seed for reproducibility
fake = Faker()
Faker.seed(12345)
random.seed(12345)

# Enhanced data definitions
departments = ['Engineering', 'Marketing', 'Sales', 'HR', 'Finance', 'IT Support', 'Operations', 'Legal', 'Research']
positions = ['Manager', 'Engineer', 'Specialist', 'Director', 'Coordinator', 'Analyst', 'Lead', 'Associate', 'Senior Engineer']
conditions = ['greater than', 'less than', 'equal to', 'not equal to', 'after', 'before']
fields = ['name', 'age', 'email', 'salary', 'department', 'position', 'hire_date', 'performance_score', 'projects_completed']
aggregations = ['COUNT', 'AVG', 'SUM', 'MAX', 'MIN']
join_types = ['INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN']
group_by_having = ['COUNT', 'AVG', 'SUM']

def generate_basic_query():
    """Generate simple SELECT-WHERE queries"""
    num_fields = random.randint(1, 4)
    selected_fields = random.sample(fields, num_fields)
    condition_field = random.choice(fields)
    condition_operator = random.choice(conditions)
    
    # Generate condition value
    condition_value = generate_field_value(condition_field)
    
    nl_query = f"Show me the {', '.join(selected_fields)} of employees where {condition_field} is {condition_operator} {condition_value}"
    
    operator_mapping = {
        'greater than': '>', 'less than': '<', 'equal to': '=',
        'not equal to': '!=', 'after': '>', 'before': '<'
    }
    sql_operator = operator_mapping[condition_operator]
    
    sql_condition = format_condition(condition_field, sql_operator, condition_value)
    sql_query = f"SELECT {', '.join(selected_fields)} FROM employees WHERE {sql_condition};"
    
    return nl_query, sql_query

def generate_join_query():
    """Generate queries with JOINs"""
    tables = ['employees e', 'departments d', 'projects p']
    join_conditions = [
        ('e.department_id = d.id', 'department'),
        ('e.id = p.employee_id', 'project')
    ]
    
    selected_tables = random.sample(tables, 2)
    join_type = random.choice(join_types)
    join_condition = next(jc for jc in join_conditions if any(t in selected_tables[1] for t in jc[1]))
    
    fields_map = {
        'e': ['name', 'salary', 'position'],
        'd': ['department_name', 'location'],
        'p': ['project_name', 'start_date']
    }
    
    selected_fields = []
    for table in selected_tables:
        table_alias = table.split()[1]
        fields = fields_map[table_alias]
        selected_fields.extend([f"{table_alias}.{f}" for f in random.sample(fields, 2)])
    
    nl_query = f"Show me {', '.join([f.split('.')[1] for f in selected_fields])} for employees and their {join_condition[1]}"
    sql_query = f"""SELECT {', '.join(selected_fields)} 
                   FROM {selected_tables[0]} 
                   {join_type} {selected_tables[1]} ON {join_condition[0]};"""
    
    return nl_query, sql_query

def generate_aggregate_query():
    """Generate queries with aggregations"""
    agg_func = random.choice(aggregations)
    group_field = random.choice(['department', 'position'])
    agg_field = random.choice(['salary', 'age'])
    
    nl_query = f"What is the {agg_func.lower()} {agg_field} for each {group_field}"
    sql_query = f"""SELECT {group_field}, {agg_func}({agg_field}) as {agg_field}_{agg_func.lower()}
                   FROM employees 
                   GROUP BY {group_field};"""
    
    return nl_query, sql_query

def generate_complex_query():
    """Generate complex queries with subqueries, multiple conditions, etc."""
    # Example: Find employees with above-average salary in their department
    nl_query = "Find employees who earn more than the average salary in their department"
    sql_query = """
    SELECT e1.name, e1.salary, e1.department
    FROM employees e1
    WHERE e1.salary > (
        SELECT AVG(e2.salary)
        FROM employees e2
        WHERE e2.department = e1.department
    );"""
    
    return nl_query, sql_query

def generate_field_value(field):
    """Generate appropriate values for different fields"""
    if field == 'department':
        return random.choice(departments)
    elif field == 'position':
        return random.choice(positions)
    elif field == 'age':
        return random.randint(22, 65)
    elif field == 'salary':
        return random.randint(40000, 150000)
    elif field == 'hire_date':
        return fake.date_between(start_date='-10y', end_date='today').strftime('%Y-%m-%d')
    elif field == 'performance_score':
        return random.randint(1, 5)
    elif field == 'projects_completed':
        return random.randint(0, 50)
    else:
        return fake.name()

def format_condition(field, operator, value):
    """Format SQL condition based on field type"""
    if field in ['age', 'salary', 'performance_score', 'projects_completed']:
        return f"{field} {operator} {value}"
    return f"{field} {operator} '{value}'"

def generate_dataset(num_samples=100000, filename='nl_sql_dataset.csv'):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['id', 'natural_language', 'sql_query'])
        writer.writeheader()
        
        # Distribution of query types
        basic_queries = int(num_samples * 0.4)  # 40%
        join_queries = int(num_samples * 0.3)   # 30%
        agg_queries = int(num_samples * 0.2)    # 20%
        complex_queries = num_samples - basic_queries - join_queries - agg_queries  # 10%
        
        id_counter = 1
        
        # Generate different types of queries
        for _ in range(basic_queries):
            nl, sql = generate_basic_query()
            writer.writerow({'id': id_counter, 'natural_language': nl, 'sql_query': sql})
            id_counter += 1
            
        for _ in range(join_queries):
            nl, sql = generate_join_query()
            writer.writerow({'id': id_counter, 'natural_language': nl, 'sql_query': sql})
            id_counter += 1
            
        for _ in range(agg_queries):
            nl, sql = generate_aggregate_query()
            writer.writerow({'id': id_counter, 'natural_language': nl, 'sql_query': sql})
            id_counter += 1
            
        for _ in range(complex_queries):
            nl, sql = generate_complex_query()
            writer.writerow({'id': id_counter, 'natural_language': nl, 'sql_query': sql})
            id_counter += 1
            
        print(f"Generated {num_samples} queries:")
        print(f"- {basic_queries} basic queries")
        print(f"- {join_queries} join queries")
        print(f"- {agg_queries} aggregate queries")
        print(f"- {complex_queries} complex queries")

if __name__ == "__main__":
    generate_dataset()