import csv
import random
import re
from faker import Faker
from datetime import datetime, timedelta
import itertools


fake = Faker()
Faker.seed(12345)
random.seed(12345)

# Enhanced schema definitions with multiple table types
SCHEMAS = {
    'employees': {
        'fields': ['id', 'name', 'age', 'email', 'salary', 'department_id', 'position', 'hire_date', 'performance_score', 'projects_completed', 'manager_id'],
        'types': {'id': 'int', 'name': 'str', 'age': 'int', 'email': 'str', 'salary': 'int', 'department_id': 'int', 'position': 'str', 'hire_date': 'date', 'performance_score': 'float', 'projects_completed': 'int', 'manager_id': 'int'}
    },
    'departments': {
        'fields': ['id', 'name', 'location', 'budget', 'manager_id'],
        'types': {'id': 'int', 'name': 'str', 'location': 'str', 'budget': 'int', 'manager_id': 'int'}
    },
    'projects': {
        'fields': ['id', 'name', 'start_date', 'end_date', 'budget', 'status', 'department_id'],
        'types': {'id': 'int', 'name': 'str', 'start_date': 'date', 'end_date': 'date', 'budget': 'int', 'status': 'str', 'department_id': 'int'}
    },
    'sales': {
        'fields': ['id', 'customer_name', 'product', 'amount', 'sale_date', 'employee_id', 'region'],
        'types': {'id': 'int', 'customer_name': 'str', 'product': 'str', 'amount': 'float', 'sale_date': 'date', 'employee_id': 'int', 'region': 'str'}
    }
}

# Diverse vocabulary for natural language queries
QUERY_STARTERS = [
    "Show me", "Find", "Get", "List", "Display", "Retrieve", "What are", "Which", 
    "Tell me", "Give me", "I need", "Can you show", "Please get", "I want to see",
    "Fetch", "Return", "Provide", "Extract", "Search for"
]

QUESTION_WORDS = ["What", "Which", "Who", "When", "Where", "How many", "How much"]

CONNECTORS = ["and", "or", "but", "also", "as well as", "plus", "along with"]

COMPARISON_PHRASES = {
    '>': ['greater than', 'more than', 'above', 'higher than', 'over', 'exceeding'],
    '<': ['less than', 'below', 'under', 'lower than', 'fewer than'],
    '=': ['equal to', 'equals', 'is', 'exactly', 'same as'],
    '!=': ['not equal to', 'different from', 'not', 'excluding'],
    'LIKE': ['contains', 'includes', 'with', 'having', 'like'],
    'IN': ['in', 'among', 'within', 'from the list'],
    'BETWEEN': ['between', 'from', 'in the range']
}

AGG_PHRASES = {
    'COUNT': ['count', 'number of', 'how many', 'total count'],
    'SUM': ['sum', 'total', 'add up', 'sum of'],
    'AVG': ['average', 'mean', 'avg'],
    'MAX': ['maximum', 'highest', 'largest', 'max', 'top'],
    'MIN': ['minimum', 'lowest', 'smallest', 'min', 'bottom']
}

class QueryGenerator:
    def __init__(self):
        self.departments = ['Engineering', 'Marketing', 'Sales', 'HR', 'Finance', 'IT', 'Operations', 'Legal', 'Research']
        self.positions = ['Manager', 'Engineer', 'Specialist', 'Director', 'Coordinator', 'Analyst', 'Lead', 'Associate', 'Senior', 'Junior']
        self.products = ['Laptop', 'Phone', 'Tablet', 'Software', 'Service', 'Subscription', 'Hardware']
        self.regions = ['North', 'South', 'East', 'West', 'Central', 'International']
        self.statuses = ['Active', 'Completed', 'Pending', 'Cancelled', 'On Hold']



    def generate_value(self, field_type, field_name=None):
        """Generate realistic values based on field type and context"""
        if field_type == 'int':
            if 'salary' in field_name.lower():
                return random.randint(30000, 200000)
            elif 'age' in field_name.lower():
                return random.randint(22, 65)
            elif 'budget' in field_name.lower():
                return random.randint(10000, 1000000)
            elif 'score' in field_name.lower():
                return round(random.uniform(1.0, 5.0), 1)
            else:
                return random.randint(1, 100)
        elif field_type == 'float':
            return round(random.uniform(100.0, 10000.0), 2)
        elif field_type == 'date':
            return fake.date_between(start_date='-5y', end_date='today').strftime('%Y-%m-%d')
        elif field_type == 'str':
            if 'department' in field_name.lower():
                return random.choice(self.departments)
            elif 'position' in field_name.lower():
                return random.choice(self.positions)
            elif 'product' in field_name.lower():
                return random.choice(self.products)
            elif 'region' in field_name.lower():
                return random.choice(self.regions)
            elif 'status' in field_name.lower():
                return random.choice(self.statuses)
            elif 'name' in field_name.lower():
                return fake.name()
            else:
                return fake.word()



    def create_natural_condition(self, field, operator, value, field_type):
        """Create natural language condition with variety"""
        comparison_phrase = random.choice(COMPARISON_PHRASES.get(operator, ['is']))
        
        if field_type == 'str' and operator == 'LIKE':
            return f"{field} {comparison_phrase} '{value}'"
        elif field_type == 'date':
            if operator in ['>', '<']:
                return f"{field} is {comparison_phrase} {value}"
            else:
                return f"{field} {comparison_phrase} '{value}'"
        elif field_type in ['int', 'float']:
            return f"{field} is {comparison_phrase} {value}"
        else:
            return f"{field} {comparison_phrase} '{value}'"



    def generate_basic_select(self):
        """Generate varied SELECT queries with different patterns"""
        table = random.choice(list(SCHEMAS.keys()))
        schema = SCHEMAS[table]
        

        num_fields = random.randint(1, min(5, len(schema['fields'])))
        if random.random() < 0.1:
            selected_fields = ['*']
            field_phrase = "all information"
        else:
            selected_fields = random.sample(schema['fields'], num_fields)
            field_phrase = ', '.join(selected_fields)
        

        starter = random.choice(QUERY_STARTERS)
        

        conditions = []
        sql_conditions = []
        
        if random.random() < 0.7:
            num_conditions = random.randint(1, 3)
            for _ in range(num_conditions):
                cond_field = random.choice(schema['fields'])
                field_type = schema['types'][cond_field]
                

                if field_type == 'str':
                    operator = random.choice(['=', '!=', 'LIKE'])
                elif field_type == 'date':
                    operator = random.choice(['>', '<', '=', 'BETWEEN'])
                else:
                    operator = random.choice(['>', '<', '=', '!=', 'BETWEEN'])
                
                value = self.generate_value(field_type, cond_field)
                

                nl_condition = self.create_natural_condition(cond_field, operator, value, field_type)
                conditions.append(nl_condition)
                

                if operator == 'LIKE':
                    sql_conditions.append(f"{cond_field} LIKE '%{value}%'")
                elif operator == 'BETWEEN' and field_type in ['int', 'float']:
                    value2 = self.generate_value(field_type, cond_field)
                    sql_conditions.append(f"{cond_field} BETWEEN {min(value, value2)} AND {max(value, value2)}")
                elif field_type == 'str':
                    sql_conditions.append(f"{cond_field} {operator} '{value}'")
                else:
                    sql_conditions.append(f"{cond_field} {operator} {value}")


        if conditions:
            condition_text = " where " + " and ".join(conditions)
        else:
            condition_text = ""
        
        nl_query = f"{starter} {field_phrase} from {table}{condition_text}"
        

        fields_sql = ', '.join(selected_fields)
        if sql_conditions:
            where_clause = " WHERE " + " AND ".join(sql_conditions)
        else:
            where_clause = ""
        
        sql_query = f"SELECT {fields_sql} FROM {table}{where_clause};"
        
        return nl_query, sql_query



    def generate_aggregate_query(self):
        """Generate diverse aggregation queries"""
        table = random.choice(list(SCHEMAS.keys()))
        schema = SCHEMAS[table]
        

        agg_func = random.choice(list(AGG_PHRASES.keys()))
        numeric_fields = [f for f, t in schema['types'].items() if t in ['int', 'float']]
        
        if agg_func == 'COUNT':
            agg_field = random.choice(['*'] + schema['fields'])
        else:
            agg_field = random.choice(numeric_fields) if numeric_fields else schema['fields'][0]

        group_by_field = None
        if random.random() < 0.6:
            categorical_fields = [f for f, t in schema['types'].items() if t == 'str']
            if categorical_fields:
                group_by_field = random.choice(categorical_fields)
        
        # Natural language construction
        agg_phrase = random.choice(AGG_PHRASES[agg_func])
        starter = random.choice(QUESTION_WORDS + QUERY_STARTERS)
        
        if group_by_field:
            nl_query = f"{starter} is the {agg_phrase} of {agg_field} for each {group_by_field} in {table}"
            sql_query = f"SELECT {group_by_field}, {agg_func}({agg_field}) FROM {table} GROUP BY {group_by_field};"
        else:
            nl_query = f"{starter} is the {agg_phrase} of {agg_field} in {table}"
            sql_query = f"SELECT {agg_func}({agg_field}) FROM {table};"
        
        return nl_query, sql_query



    def generate_join_query(self):
        """Generate realistic JOIN queries"""

        joins = [
            ('employees', 'departments', 'employees.department_id = departments.id'),
            ('employees', 'projects', 'employees.id = projects.employee_id'),
            ('sales', 'employees', 'sales.employee_id = employees.id'),
            ('projects', 'departments', 'projects.department_id = departments.id')
        ]
        
        join_info = random.choice(joins)
        table1, table2, condition = join_info
        
        # Select fields from both tables
        fields1 = random.sample(SCHEMAS[table1]['fields'], random.randint(1, 3))
        fields2 = random.sample(SCHEMAS[table2]['fields'], random.randint(1, 3))
        
        # Avoid ID conflicts
        selected_fields = []
        for field in fields1:
            if field == 'id':
                selected_fields.append(f"{table1}.{field}")
            else:
                selected_fields.append(field)
        
        for field in fields2:
            if field == 'id':
                selected_fields.append(f"{table2}.{field}")
            else:
                selected_fields.append(field)
        
        join_type = random.choice(['INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN'])
        
        # Natural language
        starter = random.choice(QUERY_STARTERS)
        nl_query = f"{starter} {', '.join([f.split('.')[-1] for f in selected_fields])} from {table1} and {table2}"
        
        # SQL query
        sql_query = f"""SELECT {', '.join(selected_fields)} 
FROM {table1} 
{join_type} {table2} ON {condition};"""
        
        return nl_query, sql_query

    def generate_complex_query(self):
        """Generate complex queries with subqueries, HAVING, etc."""
        patterns = [
            self.generate_subquery,
            self.generate_having_query,
            self.generate_multiple_table_query,
            self.generate_window_function_query
        ]
        
        return random.choice(patterns)()

    def generate_subquery(self):
        """Generate queries with subqueries"""
        templates = [
            {
                'nl': "Find employees who earn more than the average salary",
                'sql': """SELECT name, salary FROM employees 
WHERE salary > (SELECT AVG(salary) FROM employees);"""
            },
            {
                'nl': "Show departments with more than 5 employees",
                'sql': """SELECT d.name FROM departments d 
WHERE (SELECT COUNT(*) FROM employees e WHERE e.department_id = d.id) > 5;"""
            },
            {
                'nl': "Get employees working on the most expensive project",
                'sql': """SELECT e.name FROM employees e 
JOIN projects p ON e.id = p.employee_id 
WHERE p.budget = (SELECT MAX(budget) FROM projects);"""
            }
        ]
        
        template = random.choice(templates)
        return template['nl'], template['sql']

    def generate_having_query(self):
        """Generate GROUP BY ... HAVING queries"""
        table = random.choice(['employees', 'sales'])
        
        if table == 'employees':
            nl_query = "Show departments with average salary greater than 50000"
            sql_query = """SELECT department_id, AVG(salary) as avg_salary 
FROM employees 
GROUP BY department_id 
HAVING AVG(salary) > 50000;"""
        else:
            nl_query = "Find regions with total sales above 100000"
            sql_query = """SELECT region, SUM(amount) as total_sales 
FROM sales 
GROUP BY region 
HAVING SUM(amount) > 100000;"""
        
        return nl_query, sql_query

    def generate_multiple_table_query(self):
        """Generate queries involving multiple tables without explicit joins"""
        nl_query = "Show employee names and their department names"
        sql_query = """SELECT e.name, d.name as department_name 
FROM employees e, departments d 
WHERE e.department_id = d.id;"""
        
        return nl_query, sql_query

    def generate_window_function_query(self):
        """Generate queries with window functions"""
        nl_query = "Rank employees by salary within their department"
        sql_query = """SELECT name, salary, department_id,
RANK() OVER (PARTITION BY department_id ORDER BY salary DESC) as salary_rank
FROM employees;"""
        
        return nl_query, sql_query

def generate_enhanced_dataset(num_samples=100000, filename='enhanced_nl_sql_dataset.csv'):
    """Generate comprehensive NL-to-SQL dataset with realistic variety"""
    generator = QueryGenerator()
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['id', 'natural_language', 'sql_query'])
        writer.writeheader()
        
        # Query distribution
        distributions = {
            'basic_select': 0.35,   
            'aggregate': 0.25,       
            'join': 0.20,         
            'complex': 0.20       
        }
        
        id_counter = 1
        type_counts = {}
        
        for query_type, percentage in distributions.items():
            count = int(num_samples * percentage)
            type_counts[query_type] = count
            
            for _ in range(count):
                try:
                    if query_type == 'basic_select':
                        nl, sql = generator.generate_basic_select()
                        complexity = 'simple'
                    elif query_type == 'aggregate':
                        nl, sql = generator.generate_aggregate_query()
                        complexity = 'medium'
                    elif query_type == 'join':
                        nl, sql = generator.generate_join_query()
                        complexity = 'medium'
                    elif query_type == 'complex':
                        nl, sql = generator.generate_complex_query()
                        complexity = 'hard'
                    
                    writer.writerow({
                        'id': id_counter,
                        'natural_language': nl.strip(),
                        'sql_query': sql.strip()
                    })
                    id_counter += 1
                    
                except Exception as e:
                    print(f"Error generating {query_type} query: {e}")
                    continue
        
        print(f"Generated {num_samples} queries:")
        for query_type, count in type_counts.items():
            print(f"- {count} {query_type} queries")

if __name__ == "__main__":
    generate_enhanced_dataset(20000, 'nl_sql_dataset.csv')
    print("Dataset generation complete!")