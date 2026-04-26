import numpy as np
import random


# 函数定义（根据文档还原，修正格式歧义）
def fitness_function(x, y):
    # 文档函数表达式修正为：f(x,y) = 43*(1+0.125y)*cos(x/2) + 3*2.6（结合常见遗传算法测试函数逻辑）
    return 43 * (1 + 0.125 * y) * np.cos(x / 2) + 7.8


# 二进制编码与解码
def encode(x, y, x_range=[0, 10], y_range=[0, 10], bit_length=16):
    # 每个变量16位，总长度32位
    x_bits = int(((x - x_range[0]) / (x_range[1] - x_range[0])) * (2 ** bit_length - 1))
    y_bits = int(((y - y_range[0]) / (y_range[1] - y_range[0])) * (2 ** bit_length - 1))
    # 拼接二进制字符串
    return bin(x_bits)[2:].zfill(bit_length) + bin(y_bits)[2:].zfill(bit_length)


def decode(chromosome, x_range=[0, 10], y_range=[0, 10], bit_length=16):
    # 拆分x和y的编码
    x_bits = chromosome[:bit_length]
    y_bits = chromosome[bit_length:]
    # 解码为十进制
    x = x_range[0] + (int(x_bits, 2) / (2 ** bit_length - 1)) * (x_range[1] - x_range[0])
    y = y_range[0] + (int(y_bits, 2) / (2 ** bit_length - 1)) * (y_range[1] - y_range[0])
    return x, y


# 初始化种群
def init_population(pop_size, bit_length=16):
    population = []
    for _ in range(pop_size):
        # 随机生成32位二进制字符串
        chromosome = ''.join([str(random.randint(0, 1)) for _ in range(2 * bit_length)])
        population.append(chromosome)
    return population


# 选择操作（轮盘赌）
def selection(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    probabilities = [f / total_fitness for f in fitness_scores]
    # 轮盘赌选择
    selected = random.choices(population, weights=probabilities, k=len(population))
    return selected


# 交叉操作（随机单点交叉）
def crossover(population, crossover_rate=0.8):
    offspring = []
    for i in range(0, len(population), 2):
        parent1 = population[i]
        parent2 = population[i + 1] if i + 1 < len(population) else parent1
        if random.random() < crossover_rate:
            # 随机选择交叉点（排除首尾）
            cross_point = random.randint(1, len(parent1) - 1)
            child1 = parent1[:cross_point] + parent2[cross_point:]
            child2 = parent2[:cross_point] + parent1[cross_point:]
            offspring.extend([child1, child2])
        else:
            offspring.extend([parent1, parent2])
    return offspring[:len(population)]  # 保持种群规模


# 变异操作（多种方式）
def mutate(population, mutation_rate=0.01, mutation_type='uniform'):
    offspring = []
    for chromosome in population:
        if mutation_type == 'uniform':
            # 均匀变异：每个位点独立变异
            mutated = ''.join([str(1 - int(c)) if random.random() < mutation_rate else c
                               for c in chromosome])
        elif mutation_type == 'bit_flip':  # 位点变异（单点）
            mutated = list(chromosome)
            if random.random() < mutation_rate:
                pos = random.randint(0, len(chromosome) - 1)
                mutated[pos] = str(1 - int(mutated[pos]))
            mutated = ''.join(mutated)
        elif mutation_type == 'inversion':  # 逆转变异
            mutated = list(chromosome)
            if random.random() < mutation_rate:
                start = random.randint(0, len(chromosome) - 2)
                end = random.randint(start + 1, len(chromosome) - 1)
                mutated[start:end + 1] = reversed(mutated[start:end + 1])
            mutated = ''.join(mutated)
        elif mutation_type == 'swap':  # 互换变异
            mutated = list(chromosome)
            if random.random() < mutation_rate:
                pos1 = random.randint(0, len(chromosome) - 1)
                pos2 = random.randint(0, len(chromosome) - 1)
                mutated[pos1], mutated[pos2] = mutated[pos2], mutated[pos1]
            mutated = ''.join(mutated)
        else:
            mutated = chromosome
        offspring.append(mutated)
    return offspring


# 遗传算法主流程
def genetic_algorithm(pop_size=20, mutation_type='uniform', generations=100):
    bit_length = 16
    population = init_population(pop_size, bit_length)

    for _ in range(generations):
        # 计算适应度
        fitness_scores = []
        for chrom in population:
            x, y = decode(chrom, bit_length=bit_length)
            fitness = fitness_function(x, y)
            fitness_scores.append(fitness)

        # 选择、交叉、变异
        selected = selection(population, fitness_scores)
        crossed = crossover(selected)
        population = mutate(crossed, mutation_type=mutation_type)

    # 最终代适应度计算
    final_fitness = []
    final_individuals = []
    for chrom in population:
        x, y = decode(chrom, bit_length=bit_length)
        fit = fitness_function(x, y)
        final_fitness.append(fit)
        final_individuals.append((x, y))

    # 统计结果
    best_idx = np.argmax(final_fitness)
    worst_idx = np.argmin(final_fitness)
    return {
        'best_fitness': round(final_fitness[best_idx], 4),
        'avg_fitness': round(np.mean(final_fitness), 4),
        'worst_fitness': round(final_fitness[worst_idx], 4),
        'best_individual': (round(final_individuals[best_idx][0], 4), round(final_individuals[best_idx][1], 4)),
        'worst_individual': (round(final_individuals[worst_idx][0], 4), round(final_individuals[worst_idx][1], 4))
    }


# 批量运行测试（填充表格）
if __name__ == '__main__':
    # 表一：默认操作（种群20，均匀变异）
    print("表一：默认操作下的适应度对比")
    default_result = genetic_algorithm(pop_size=20, mutation_type='uniform')
    print(f"最佳适应度：{default_result['best_fitness']}")
    print(f"平均适应度：{default_result['avg_fitness']}")
    print(f"最差适应度：{default_result['worst_fitness']}")
    print(f"最佳个体(x,y)：{default_result['best_individual']}")
    print(f"最差个体(x,y)：{default_result['worst_individual']}\n")

    # 表二：种群规模改变
    print("表二：种群规模改变下的适应度对比")
    pop_sizes = [5, 20, 100]
    for pop in pop_sizes:
        result = genetic_algorithm(pop_size=pop, mutation_type='uniform')
        print(f"种群数量{pop}：")
        print(f"  最佳适应度：{result['best_fitness']}，平均适应度：{result['avg_fitness']}，最差适应度：{result['worst_fitness']}")
        print(f"  最佳个体：{result['best_individual']}，最差个体：{result['worst_individual']}\n")

    # 表三：变异操作改变
    print("表三：变异操作改变下的适应度对比")
    mutation_types = ['bit_flip', 'inversion', 'swap', 'uniform']
    for mt in mutation_types:
        result = genetic_algorithm(pop_size=20, mutation_type=mt)
        print(f"变异操作{mt}：")
        print(f"  最佳适应度：{result['best_fitness']}，平均适应度：{result['avg_fitness']}，最差适应度：{result['worst_fitness']}")
        print(f"  最佳个体：{result['best_individual']}，最差个体：{result['worst_individual']}\n")


