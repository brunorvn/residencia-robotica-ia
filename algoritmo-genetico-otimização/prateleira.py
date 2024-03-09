import random
from deap import base, creator, tools, algorithms

PRODUCTS = [
    [1, 50, 2, 65],
    [2, 35, 2, 45],
    [3, 25, 3, 58],
    [4, 20, 4, 71],
    [5, 45, 4, 71],
    [6, 50, 6, 77],
    [7, 45, 5, 90],
    [8, 40, 5, 90],
    [9, 30, 6, 65],
    [10, 50, 4, 52],
    [11, 35, 2, 90],
    [12, 50, 6, 52],
    [13, 20, 5, 71],
    [14, 25, 3, 77],
    [15, 30, 4, 58],
    [16, 20, 2, 45],
    [17, 60, 2, 65],
    [18, 35, 1, 103],
    [19, 25, 5, 71],
    [20, 45, 4, 97],
]

class Product(object):
    def __init__(self, id, available_stock, unit_profit, unit_cm):
        self.id = id
        self.stock = available_stock
        self.unitary_profit = unit_profit
        self.size = unit_cm

def generate_individual():
    return random.sample(range(len(PRODUCTS)), len(PRODUCTS))

def evaluate_individual(individual):
    shelf = [0] * len(PRODUCTS)
    total_profit = 0
    total_size = 0

    for product_index in individual:
        product = PRODUCTS[product_index]
        if shelf[product_index] < product[1]:  # Check if there is stock available
            if total_size + product[3] <= 37200:  # Check if it fits on the shelf
                total_profit += product[2]
                total_size += product[3]
                shelf[product_index] += 1

    return (total_profit,)

def main():
    
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("individual", generate_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)  
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.5)
    toolbox.register("select", tools.selTournament, tournsize=3)        
    
    random.seed(42)

    population = toolbox.population(n=100)

    hall_of_fame = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", max)

    algorithms.eaSimple(
        population,
        toolbox,
        cxpb=0.7,
        mutpb=0.2,
        ngen=100,
        stats=stats,
        halloffame=hall_of_fame,
        verbose=True,
    )

    best_individual = hall_of_fame[0]
    print("Best Shelf Arrangement: ", best_individual)
    best_profit = evaluate_individual(best_individual)[0]
    print("Best Profit: ", best_profit)
    
if __name__ == "__main__":
    main()
    
    
