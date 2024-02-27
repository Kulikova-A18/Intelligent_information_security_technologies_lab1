import math
import matplotlib.pyplot as plt
import itertools

NORM_LEARNING = 0.3
RFB_NETS = [[0, 0, 0, 0], [0, 0, 1, 1], [1, 0, 0, 0]]

# Создается класс Network, который содержит методы и переменные для обучения и использования нейронной сети
class Network:
    def __init__(self, synapses: list, RBF: list, norm_learn: int):
        # Инициализация параметров нейронной сети
        self.synapses = synapses  # Весовые коэффициенты
        self.epoch_rate = 0  # Номер эпохи
        self.input_vec = input_vector()  # Входной вектор
        self.RBF = RBF  # RBF (Radial Basis Function)
        self.norm_learn = norm_learn  # Норма обучения
        self.errors = []  # Список для хранения ошибок на каждой эпохе

        # Вывод информации о созданной нейронной сети
        print("Создана нейронная сеть со следующими параметрами: ")
        print("Номер Эпохи: ", self.epoch_rate)
        print("Вектор весов: ", self.synapses)
        print("Выходной вектор: ", self.out_vec())
        print("Суммарная ошибка: ", self.err())
        print("НОрма ообучения: ", self.norm_learn)

    def func_activation(self, net):
        if net >= 0:
            return 1
        else:
            return 0

    def func_activation_2(self, net):
        return 1/2 * (net / (1 + abs(net)) + 1)

    # вычисляется радиально-базисная функция для входных данных
    def find_fi(self, inputs: list):
        fi = []
        for i in range(len(self.RBF)):
            buf = 0
            for j in range(1, 5):
                buf += (inputs[j] - self.RBF[i][j - 1]) ** 2
            fi.append(math.exp(-buf))
        return fi

    # выполняет итерацию нейронной сети для входных данных
    def itertion(self, inputs: list):  # итерация нейронной сети по входным данным
        net = 0
        fi = self.find_fi(inputs)
        for i in range(len(fi)):
            net += self.synapses[i + 1] * fi[i]
        # return self.func_activation(net + self.synapses[0])
        return self.func_activation_2(net + self.synapses[0])

    # выполняет одну эпоху обучения нейронной сети
    def epoch_1(self, out=True):
        for i in range(len(self.input_vec)):
            self.step_learning_1((self.input_vec[i]))
        self.epoch_rate += 1
        if out:
            print()
            print("Номер Эпохи: ", self.epoch_rate)
            print("Вектор весов: ", [round(elem, 2) for elem in self.synapses])
            print("Выходной вектор: ", self.out_vec())
            print("Суммарная ошибка: ", self.err())

        self.errors.append(self.err())

    # выполняет шаг обучения нейронной сети
    def step_learning_1(self, inputs: list):
        buf = function(inputs) - self.itertion(inputs)
        net = 0
        fi = self.find_fi(inputs)
        for i in range(len(self.synapses) - 1):
            net += self.synapses[i + 1] * fi[i]
        net += self.synapses[0]

        for i in range(len(self.RBF)):
            self.synapses[i + 1] += self.norm_learn * buf * fi[i]
        self.synapses[0] += self.norm_learn * buf
        return 1

    # для сброса сети, вывода результата, подсчета ошибки и поиска оптимального
    # подмножества входных данных соответственно
    def reset_net(self):
        self.synapses = [0, 0, 0, 0]
        self.epoch_rate = 0
        self.error_count = []
        self.error_count.append(self.err())

    # для сброса сети, вывода результата, подсчета ошибки и поиска оптимального
    # подмножества входных данных соответственно
    def out_vec(self):
        buf = []
        input_vec = input_vector()
        for i in range(len(input_vec)):
            if self.itertion(input_vec[i]) >= 0.5:
                buf.append(1)
            else:
                buf.append(0)
        return buf

    def err(self):
        err = 0
        input_vec = input_vector()
        for i in range(len(input_vec)):
            if self.out_vec()[i] != function(input_vec[i]): err += 1
        return err

    # для сброса сети, вывода результата, подсчета ошибки и поиска оптимального
    # подмножества входных данных соответственно
    def find_subset(self):
        out = input_vector()
        for i in range(1, 16, 1):
            subset = list(itertools.combinations(out, i))
            print("Размер подмножества: ", len(subset))
            for j in subset:
                self.reset_net()
                self.input_vec = j
                k = 0
                while (self.err() > 0 and k < 20):
                    self.epoch_1(False)
                    k += 1
                if (self.err() == 0):
                    buf = j
                    print("Новое подмножество найдено: {}".format(j))
                    print("Количество эпох: {}".format(self.epoch_rate))
                    print()
                    return buf
                    break
        return buf

# определена логическая функция, которая вычисляется на входных данных
def function(x: list):
    # (!x1 + x3) * x2 + x2 * x4
    return int(((not x[1] or x[3]) and x[2]) or (x[2] and x[4]))

# создается список всех возможных комбинаций входных значений для заданной логической функции
def input_vector():
    out = []
    for i in range(16):
        out.append([])
        buf = i
        for j in range(4):
            out[i].insert(0, buf % 2)
            buf = int(buf / 2)
        out[i].insert(0, 1)
    return out


def main():
    Net = Network([0, 0, 0, 0], RFB_NETS, NORM_LEARNING)
    i = 0
    while Net.err() > 0:
        Net.epoch_1()
        i += 1

    x = [i for i in range(1, len(Net.errors) + 1)]
    y = Net.errors
    plt.plot(x, y)
    plt.plot(x, y, 'bo')
    plt.show()


if __name__ == "__main__":
    main()
