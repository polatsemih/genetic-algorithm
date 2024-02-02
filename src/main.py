# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import random
import time

# A seed for reproducibility
seed = 1234
random.seed(seed)

class GeneticAlgorithm:
    def __init__(self, dataset_path, population_number, individual_length, termination_conditions,
                 tournament_selection_k_number, crossover_probability, mutation_probability, results_path):
        self.df = pd.read_csv(dataset_path)
        self.population_number = population_number
        self.individual_length = individual_length
        self.termination_conditions = termination_conditions
        self.tournament_selection_k_number = tournament_selection_k_number
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.results_path = results_path

        self.best_individuals = []
        
        # Select features from the same generated classifier model
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=seed)
        
        self.crossover_count = 0
        self.mutation_count = 0

    def create_individual(self):
        while True:
            individual = ''.join(random.choice('01') for _ in range(self.individual_length))
            if '1' in individual: # Ensure at least one feature is selected
                return individual

    def initialize_population(self):
        population = [(self.create_individual(), 0) for _ in range(self.population_number)]
        return population

    def calculate_population_fitness(self, population):
        updated_population = []
    
        for individual, fitness in population:
            fitness = self.calculate_fitness(individual)
            updated_population.append((individual, fitness))
    
        return updated_population

    def one_point_crossover(self, parents):
        parent1, parent2 = parents[0], parents[1]
        parent1, parent1_fitness = parent1[0], parent1[1]
        parent2, parent2_fitness = parent2[0], parent2[1]
        
        while True:
            crossover_point = random.randint(1, self.individual_length - 1)

            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]

            if '1' in child1 and '1' in child2: # Ensure at least one feature is selected
                best_parent, best_parent_fitness, worse_parent, worse_parent_fitness = (parent1, parent1_fitness, parent2, parent2_fitness) if parent1_fitness > parent2_fitness else (parent2, parent2_fitness, parent1, parent1_fitness)
                
                child1_fitness = self.calculate_fitness(child1)
                child2_fitness = self.calculate_fitness(child2)
                best_child, best_child_fitness = (child1, child1_fitness) if child1_fitness > child2_fitness else (child2, child2_fitness)
                
                return best_parent, best_parent_fitness, worse_parent, worse_parent_fitness, best_child, best_child_fitness

    def bitwise_mutation(self, individual):
        while True:
            mutated_individual = ''
            for bit in individual:
                if random.random() < self.mutation_probability:
                    mutated_bit = '0' if bit == '1' else '1'
                else:
                    mutated_bit = bit
                mutated_individual += mutated_bit

            if '1' in mutated_individual:  # Ensure at least one feature is selected
                mutated_individual_fitness = self.calculate_fitness(mutated_individual)
                return mutated_individual, mutated_individual_fitness

    def calculate_fitness(self, individual):
        # Get column names
        selection_indices = [int(bit) for bit in individual]
        selection_indices.append(1)  # Ensure the last column ("Outcome") is always included
        selected_columns = [col for col, select in zip(self.df.columns, selection_indices) if select]
        selected_df = self.df[selected_columns]

        # Split the dataset
        X = selected_df.drop(columns=['Outcome'])  # Features
        Y = selected_df['Outcome']  # Class
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)

        # Find fitness value
        self.rf_model.fit(X_train, Y_train)
        Y_pred = self.rf_model.predict(X_test)
        fitness = accuracy_score(Y_test, Y_pred)

        return fitness

    def run_genetic_algorithm(self, termination_condition):
        print(f"\nRunning with {termination_condition} tests...\n")
        start_time_for_run = time.time()

        # Initialize the population
        population = self.initialize_population()
        
        # Fitness calculation for population
        population = self.calculate_population_fitness(population)

        self.crossover_count = 0
        self.mutation_count = 0
        best_individuals_for_run = []
        for i in range(termination_condition):
            print(f"Running {i + 1}. test")

            is_crossover = random.random() < self.crossover_probability
            is_mutation = random.random() < self.mutation_probability
            
            if is_crossover:
                self.crossover_count += 1
                # Tournament selection
                parents = []
                for _ in range(2):
                    selected_individuals = random.sample(population, self.tournament_selection_k_number)
                    # Prioritize the highest fitness values first, and in case of ties, 
                    # select the individual with the fewest '1's in the binary string
                    parent = max(selected_individuals, key=lambda x: (x[1], -x[0].count('1')))
                    parents.append(parent)

                # Crossover
                best_parent, best_parent_fitness, worse_parent, worse_parent_fitness, best_child, best_child_fitness = self.one_point_crossover(parents)
                population.remove((worse_parent, worse_parent_fitness))
                population.append((best_child, best_child_fitness))

            # Mutation
            if is_mutation:
                self.mutation_count += 1
                random_individual = random.choice(population)
                mutated_individual, mutated_individual_fitness = self.bitwise_mutation(random_individual[0])
                population.remove((random_individual[0], random_individual[1]))
                population.append((mutated_individual, mutated_individual_fitness))
            
            # If first iteration or crossover or mutation occures select best individual else pass to next test
            if i == 0 or is_crossover or is_mutation:
                # Best individual and its fitness value for test
                best_individual_for_test, best_fitness_for_test = max(population, key=lambda x: (x[1], -x[0].count('1')))

            best_individuals_for_run.append((best_individual_for_test, best_fitness_for_test))

        end_time_for_run = time.time()
        total_time_for_run = end_time_for_run - start_time_for_run

        self.save_results_from_tests(termination_condition, best_individuals_for_run, total_time_for_run)

    def save_results_from_tests(self, termination_condition, best_individuals_for_run, total_time_for_run):
        best_individual_for_run = max(best_individuals_for_run, key=lambda x: (x[1], -x[0].count('1')))
        self.best_individuals.append(best_individual_for_run)

        with open(f'{self.results_path}{termination_condition}.txt', "w") as result_txt:
            result_txt.write(f"Execution time: {total_time_for_run} seconds\n\n")
            result_txt.write(f"Number of crossovers applied: {self.crossover_count}\n")
            result_txt.write(f"Number of mutations applied: {self.mutation_count}\n\n")
            result_txt.write("Best individuals and their accuracies for each test:\n")

            for j, (best_individual_for_test, best_fitness_for_test) in enumerate(best_individuals_for_run, start=1):
                result_txt.write(f"Test {j} => Best Individual: {best_individual_for_test}, Fitness: {best_fitness_for_test}\n")

            result_txt.write(f"\nFrom run with {termination_condition} tests:\n")
            result_txt.write(f"Best Individual: {best_individual_for_run[0]}, Fitness: {best_individual_for_run[1]}")

    def save_results_from_runs(self):
        best_individual = max(self.best_individuals, key=lambda x: (x[1], -x[0].count('1')))
        individual_with_all_features = '11111111'
        fitness_with_all_features = self.calculate_specific_fitness(individual_with_all_features)
        with open(f'{self.results_path}all.txt', "w") as result_txt:
            result_txt.write("From All Runs:\n")
            for j, (best_individual_for_run, best_fitness_for_run) in enumerate(self.best_individuals, start=1):
                result_txt.write(
                    f"Run {j} => Best Individual: {best_individual_for_run}, Fitness: {best_fitness_for_run}\n")

            result_txt.write("\nBest overall individual (S vector) and fitness value:\n")
            result_txt.write(f"Best Individual: {best_individual[0]}, Fitness: {best_individual[1]}\n\n")
            
            result_txt.write("Individual with all features selected scenario (For compare purpose):\n")
            result_txt.write(f"Individual: {individual_with_all_features}, Fitness: {fitness_with_all_features}")
            
    def calculate_specific_fitness(self, specific_individual):
        specific_fitness = self.calculate_fitness(specific_individual)
        return specific_fitness

def main():
    # Paths
    dataset_path = 'path/to/Pima Indians Diabetes Database/diabetes.csv'
    results_path = 'path/to/results/run_'

    # Constants
    population_number = 50
    individual_length = 8
    termination_conditions = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]

    tournament_selection_k_number = 5
    crossover_probability = 0.6
    mutation_probability = 0.05

    genetic_algorithm = GeneticAlgorithm(dataset_path, population_number, individual_length, termination_conditions,
                                         tournament_selection_k_number, crossover_probability, mutation_probability, results_path)

    for termination_condition in termination_conditions:
        genetic_algorithm.run_genetic_algorithm(termination_condition)

    genetic_algorithm.save_results_from_runs()

if __name__ == "__main__":
    main()