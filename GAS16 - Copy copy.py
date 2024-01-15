import pandas as pd
from datetime import datetime, timedelta
import random
from deap import base, tools, creator, algorithms
import plotly.express as px
import matplotlib.pyplot as plt

# Load data from CSV
data = pd.read_csv('GAS3.csv')
data.set_index("TaskID", inplace=True)

# Exclude TaskID 28 from the dataset
data = data.drop(index=28, errors='ignore')

# Declare variables
start_date = datetime(2023, 8, 3)
manual_chromosome = [12, 21, 5, 26, 2, 3, 9, 25, 1, 0, 10, 16, 6, 23, 13, 7, 17, 4, 19, 14, 15, 8, 18, 20, 11, 24, 27, 22]

# Define genetic algorithm parameters
population_size = 100
generations = 80
crossover_prob = 0.7
mutation_prob = 0.3

# Create a fitness class for minimizing lateness cost
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Initialize genetic algorithm operators
toolbox = base.Toolbox()
toolbox.register("evaluate", lambda ind: (sum(data.loc[i, 'CostOfLateness'] for i in ind),))
toolbox.register("indices", random.sample, range(len(data)), len(data))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Function to manually add chromosome
def add_manual_chromosome(chromosome, population):
    population.append(creator.Individual(chromosome))

def calculate_start_date(ind, task_dates, data):
    machine_start_dates = {}  # Dictionary to store the latest start date for each MachineID

    for i in range(1, len(ind)):
        machine_id = data['MachineID'][ind[i]]

        if pd.notna(data['Predecessors'][ind[i]]):
            # Extract predecessor information
            predecessor_info = data['Predecessors'][ind[i]].split(':')
            pred_id = int(predecessor_info[0])
            pred_type = predecessor_info[1]
            pred_offset = int(predecessor_info[2])

            while len(task_dates) <= pred_id:
                # Ensure the task_dates list is long enough
                task_dates.append(task_dates[-1])

            if pred_type == 'start':
                # Start date is the start date of the predecessor plus the offset
                task_start = task_dates[pred_id] + timedelta(days=pred_offset)
            else:
                # Start date is the end date of the predecessor plus the offset
                task_start = task_dates[pred_id] + timedelta(days=int(data['ProcessingTime'][pred_id])) + timedelta(days=pred_offset)
        else:
            # If no predecessor, set the start date as the end date of the previous task with the same MachineID
            last_machine_start_date = machine_start_dates.get(machine_id, start_date)

            # Set the start date as the maximum of the last machine's start date and the end date of the previous task
            task_start = max(last_machine_start_date, task_dates[i - 1])

        # Check and avoid having the same start date
        while task_start in task_dates:
            task_start += timedelta(days=1)

        task_dates.append(task_start)
        machine_start_dates[machine_id] = task_start + timedelta(days=int(data['ProcessingTime'][ind[i]]))


# Function to run the genetic algorithm
def run_genetic_algorithm():
    # Generate a random population
    population = toolbox.population(n=population_size)

    # Add manual chromosome to the population
    add_manual_chromosome(manual_chromosome, population)

    # Track all generations and their fitness values
    all_generations = []
    all_fitnesses = []

    # Evaluate the entire initial population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Save initial generation
    all_generations.append(population[:])
    all_fitnesses.append(fitnesses[:])

    # Run the genetic algorithm
    for gen in range(generations):
        offspring = algorithms.varAnd(population, toolbox, cxpb=crossover_prob, mutpb=mutation_prob)
        
        # Evaluate offspring
        fitnesses = list(map(toolbox.evaluate, offspring))
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        # Select the next generation
        population[:] = tools.selBest(population + offspring, k=population_size)

        # Save current generation
        all_generations.append(population[:])
        all_fitnesses.append(fitnesses[:])

    # Extract the best individual from the final generation
    best_individual = tools.selBest(population, k=1)[0]
    print("Best Individual:", best_individual)
    print("Total Cost of Lateness:", best_individual.fitness.values[0])

    # Generate dates for the best individual
    task_dates = [start_date]

    # Calculate start dates based on predecessor rule
    calculate_start_date(best_individual, task_dates, data)

    print("\nTask Dates:")
    for task_id, task_date in zip(best_individual, task_dates):
        print(f"Task {task_id}: {task_date.strftime('%Y-%m-%d')}")

    # Create a DataFrame for Gantt chart
    df = pd.DataFrame({
        'Task': [f'Task {i}' for i in best_individual if i in data.index],
        'Start': [task_dates[i] for i in best_individual if i in data.index],
        'End': [task_dates[i] + timedelta(days=int(data['ProcessingTime'][i])) for i in best_individual if i in data.index]
    })

    # Save all generations and fitness values to CSV
    save_generations_csv(all_generations, all_fitnesses)

    # Save Gantt chart DataFrame as a CSV file
    gantt_chart_csv_path = 'gantt_chart.csv'
    df.to_csv(gantt_chart_csv_path, index=False)
    print(f'Gantt chart data saved to {gantt_chart_csv_path}')

    # Plot Gantt chart
    fig = px.timeline(df, x_start='Start', x_end='End', y='Task', title="Gantt Chart")
    fig.update_yaxes(categoryorder='total ascending')
    fig.show()

# Function to save all generations and fitness values to CSV
def save_generations_csv(all_generations, all_fitnesses):
    data_to_save = []
    for gen, fitness_values in enumerate(all_fitnesses):
        for ind, fitness in zip(all_generations[gen], fitness_values):
            data_to_save.append([gen, ind, fitness[0]])

    df_generations = pd.DataFrame(data_to_save, columns=['Generation', 'Individual', 'Fitness'])
    generations_csv_path = 'generations_data.csv'
    df_generations.to_csv(generations_csv_path, index=False)
    print(f'Generations data saved to {generations_csv_path}')

    # Ekstrak nilai fitness untuk semua generasi
    fitness_values = [[ind.fitness.values[0] for ind in gen] for gen in all_generations]

    # Buat figure dan subplots
    fig, ax = plt.subplots()

    # Plot nilai fitness setiap individu di setiap generasi
    for i in range(max(len(gen) for gen in fitness_values)):
        ax.plot([gen[i] if i < len(gen) else None for gen in fitness_values], label=f'Individual {i+1}')

    # Atur judul dan label
    ax.set_title('Evolusi Fitness')
    ax.set_xlabel('Generasi')
    ax.set_ylabel('Fitness')

    # Tampilkan plot
    plt.show()
# Run the genetic algorithm
run_genetic_algorithm()


