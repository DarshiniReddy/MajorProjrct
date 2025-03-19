import streamlit as st
import pandas as pd
import numpy as np
import uuid
import random

# Define the Streamlit web application
st.set_page_config(page_title="Priority-based Task Scheduling", layout="centered")

# Global Data Storage
if "task_list" not in st.session_state:
    st.session_state.task_list = pd.DataFrame(columns=[
        "TaskID", "Deadline", "Computational Complexity", "User Priority", 
        "System Priority", "CPU", "RAM", "Network Bandwidth"
    ])

if "results" not in st.session_state:
    st.session_state.results = pd.DataFrame(columns=[
        "Scheduling Algorithm Used", "Avg Turnaround Time", "Avg Waiting Time"
    ])

# --- Task Creation Section ---
st.header("Create Tasks")

# Auto-generated Task ID
task_id = str(uuid.uuid4())[:8]

# Task Input Fields
col1, col2 = st.columns(2)
deadline = col1.number_input("Deadline (ms)", min_value=1, value=1000, step=100)
complexity = col2.selectbox("Computational Complexity", ["Heavy", "Medium", "Light"])
user_priority = col1.selectbox("User Defined Priority", ["High", "Medium", "Low"])

# Resource Requirements
cpu = st.number_input("CPU (1-100)", min_value=1, max_value=100, value=10, step=1)
ram = st.number_input("RAM (KB)", min_value=1, max_value=1024, value=256, step=1)
bandwidth = st.number_input("Network Bandwidth (KB, 0-10GB)", min_value=0, max_value=10_000_000, value=500_000, step=1)

# Submit New Task
if st.button("Submit Task"):
    new_task = pd.DataFrame([{
        "TaskID": task_id,
        "Deadline": deadline,
        "Computational Complexity": complexity,
        "User Priority": user_priority,
        "System Priority": None,  # To be determined
        "CPU": cpu,
        "RAM": ram,
        "Network Bandwidth": bandwidth
    }])
    st.session_state.task_list = pd.concat([st.session_state.task_list, new_task], ignore_index=True)
    st.rerun()

# Random Process Generator
def generate_random_tasks():
    complexities = ["Heavy", "Medium", "Light"]
    priorities = ["High", "Medium", "Low"]
    
    random_tasks = pd.DataFrame([
        {
            "TaskID": str(uuid.uuid4())[:8],
            "Deadline": np.random.randint(500, 5000),
            "Computational Complexity": np.random.choice(complexities),
            "User Priority": np.random.choice(priorities),
            "System Priority": None,
            "CPU": np.random.randint(1, 101),
            "RAM": np.random.randint(1, 1025),
            "Network Bandwidth": np.random.randint(0, 10_000_001)
        }
        for _ in range(30)
    ])
    
    st.session_state.task_list = pd.concat([st.session_state.task_list, random_tasks], ignore_index=True)

if st.button("Generate 30 Random Tasks"):
    generate_random_tasks()
    st.rerun()

# --- Task Table Section ---
st.header("Task List")
st.write("List of tasks created:")

if not st.session_state.task_list.empty:
    st.dataframe(st.session_state.task_list)

# --- Scheduling Algorithm Implementation ---
def heuristic_scheduling(tasks):
    """ Placeholder function for heuristic scheduling algorithms. 
    You can replace this with your own heuristic approach. """

    # Ensure 'Deadline' is numeric
    tasks["Deadline"] = pd.to_numeric(tasks["Deadline"], errors="coerce")  # Convert to numeric, replace errors with NaN
    tasks["CPU"] = pd.to_numeric(tasks["CPU"], errors="coerce")  # Convert to numeric, replace errors with NaN
    tasks["RAM"] = pd.to_numeric(tasks["RAM"], errors="coerce")  # Convert to numeric, replace errors with NaN
    tasks["Network Bandwidth"] = pd.to_numeric(tasks["Network Bandwidth"], errors="coerce")  # Convert to numeric, replace errors with NaN
    # tasks = tasks.dropna()  # Remove rows with NaN 
    # print("this is tasks", tasks)
    results = execute_scheduling(tasks)
    # print(results)
    return results
    
# Min-Min Scheduling
def min_min_scheduling(tasks):
    if tasks.empty:
        return pd.DataFrame(columns=["Scheduling Algorithm Used", "Avg Turnaround Time", "Avg Waiting Time"])
    
    tasks = tasks.copy()
    completion_time = 0
    turnaround_times = []
    waiting_times = []
    task_execution_order = []

    while not tasks.empty:
        min_task = tasks.loc[tasks["Deadline"].idxmin()]
        task_execution_order.append(min_task["TaskID"])
        execution_time = min_task["Deadline"]
        completion_time += execution_time
        turnaround_time = completion_time
        waiting_time = turnaround_time - execution_time
        turnaround_times.append(turnaround_time)
        waiting_times.append(waiting_time)
        tasks = tasks.drop(min_task.name)
    
    avg_turnaround_time = np.mean(turnaround_times)
    avg_waiting_time = np.mean(waiting_times)
    
    results = pd.DataFrame([{
        "Scheduling Algorithm Used": "Min-Min Scheduling",
        "Avg Turnaround Time": round(avg_turnaround_time, 2),
        "Avg Waiting Time": round(avg_waiting_time, 2),
        "Task Execution Order": task_execution_order
    }])
    
    return results

# Max-Min Scheduling
def max_min_scheduling(tasks):
    if tasks.empty:
        return pd.DataFrame(columns=["Scheduling Algorithm Used", "Avg Turnaround Time", "Avg Waiting Time"])
    
    tasks = tasks.copy()
    completion_time = 0
    turnaround_times = []
    waiting_times = []
    task_execution_order = []
    
    while not tasks.empty:
        max_task = tasks.loc[tasks["Deadline"].idxmax()]
        task_execution_order.append(max_task["TaskID"])
        execution_time = max_task["Deadline"]
        completion_time += execution_time
        turnaround_time = completion_time
        waiting_time = turnaround_time - execution_time
        turnaround_times.append(turnaround_time)
        waiting_times.append(waiting_time)
        tasks = tasks.drop(max_task.name)
    
    avg_turnaround_time = np.mean(turnaround_times)
    avg_waiting_time = np.mean(waiting_times)
    
    results = pd.DataFrame([{
        "Scheduling Algorithm Used": "Max-Min Scheduling",
        "Avg Turnaround Time": round(avg_turnaround_time, 2),
        "Avg Waiting Time": round(avg_waiting_time, 2),
        "Task Execution Order": task_execution_order
    }])
    
    return results

#genetic_algorithm

def genetic_algorithm_scheduling(tasks, population_size=10, generations=50, mutation_rate=0.1):
    if tasks.empty:
        return pd.DataFrame(columns=["Scheduling Algorithm Used", "Avg Turnaround Time", "Avg Waiting Time"])

    tasks = tasks.copy()
    num_tasks = len(tasks)
    
    # Generate initial population (random permutations)
    population = [list(np.random.permutation(tasks.index)) for _ in range(population_size)]

    def fitness(schedule):
        completion_time = 0
        turnaround_times = []
        waiting_times = []
        for task_idx in schedule:
            execution_time = tasks.loc[task_idx, "Deadline"]
            completion_time += execution_time
            turnaround_times.append(completion_time)
            waiting_times.append(completion_time - execution_time)
        return np.mean(turnaround_times) + np.mean(waiting_times)  # Minimize total time

    for _ in range(generations):
        # Evaluate fitness
        scores = [(schedule, fitness(schedule)) for schedule in population]
        scores.sort(key=lambda x: x[1])  # Sort by best fitness (lower is better)

        # Select the best half
        parents = [s[0] for s in scores[:population_size // 2]]

        # Crossover (two-point crossover)
        offspring = []
        for _ in range(population_size // 2):
            p1, p2 = random.sample(parents, 2)
            cut1, cut2 = sorted(random.sample(range(num_tasks), 2))
            child = p1[:cut1] + p2[cut1:cut2] + p1[cut2:]
            offspring.append(child)

        # Mutation (swap mutation)
        for child in offspring:
            if random.random() < mutation_rate:
                i, j = random.sample(range(num_tasks), 2)
                child[i], child[j] = child[j], child[i]

        # New population
        population = parents + offspring

    # Best schedule from final generation
    best_schedule = min(population, key=fitness)
    
    # Compute results
    completion_time = 0
    turnaround_times = []
    waiting_times = []
    task_execution_order = [tasks.loc[idx, "TaskID"] for idx in best_schedule]

    for task_idx in best_schedule:
        execution_time = tasks.loc[task_idx, "Deadline"]
        completion_time += execution_time
        turnaround_times.append(completion_time)
        waiting_times.append(completion_time - execution_time)

    avg_turnaround_time = np.mean(turnaround_times)
    avg_waiting_time = np.mean(waiting_times)

    results = pd.DataFrame([{
        "Scheduling Algorithm Used": "Genetic Algorithm Scheduling",
        "Avg Turnaround Time": round(avg_turnaround_time, 2),
        "Avg Waiting Time": round(avg_waiting_time, 2),
        "Task Execution Order": task_execution_order
    }])

    return results

# Function to execute scheduling algorithms
def execute_scheduling(tasks):
  #  heuristic_result = heuristic_scheduling(tasks)
    min_min_result = min_min_scheduling(tasks)
    max_min_result = max_min_scheduling(tasks)
    genetic_result = genetic_algorithm_scheduling(tasks)
    
    results = pd.concat([min_min_result, max_min_result, genetic_result], ignore_index=True)
    
    return results

# Submit Tasks Button
if st.button("Submit Tasks for Scheduling"):
    st.session_state.results = heuristic_scheduling(st.session_state.task_list)
    st.rerun()

# --- Scheduling Results Section ---
st.header("Scheduling Results")
st.write("Results from the heuristic scheduling algorithm:")

if not st.session_state.results.empty:
    st.dataframe(st.session_state.results)