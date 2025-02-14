import sqlite3
import random
from faker import Faker
from datetime import datetime

# Initialize Faker
fake = Faker()

# Connect to SQLite database
conn = sqlite3.connect('data/users.db')
cursor = conn.cursor()

# Create a table named 'users'
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    age INTEGER,
    email TEXT,
    salary REAL,
    department TEXT,
    position TEXT,
    hire_date TEXT
)
''')

# Departments and positions for sample data
departments = ['Engineering', 'Marketing', 'Sales', 'HR', 'Finance', 'IT Support']
positions = ['Manager', 'Engineer', 'Specialist', 'Director', 'Coordinator']

# Generate sample data
sample_data = []
for _ in range (10000):
    name = fake.name()
    age = random.randint(22, 65)
    email = fake.email()
    salary = round(random.uniform(40000, 150000), 2)
    department = random.choice(departments)
    position = random.choice(positions)
    hire_date = fake.date_between(start_date='-10y', end_date='today').strftime('%Y-%m-%d')
    sample_data.append((name, age, email, salary, department, position, hire_date))

# Insert sample data into the 'users' table
cursor.executemany('''
    INSERT INTO users (name, age, email, salary, department, position, hire_date)
    VALUES (?, ?, ?, ?, ?, ?, ?)
''', sample_data)

conn.commit()
conn.close()
