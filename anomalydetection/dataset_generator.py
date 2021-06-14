import datetime
import random
import matplotlib.pyplot as plt
import pandas as pd


class Generator:

    def __init__(self, start, end):
        self.start = start
        self.end = end

    def generate_with_outliers(self, records, filename):
        self.range_outliers = 200
        now = datetime.datetime.now()
        iter_range = iter(range(records))
        with open(filename, 'w') as file:
            for i in iter_range:
                if random.choice(range(1000)) == 9:
                    multiplier = random.randint(-5, 5) or 6
                    for j in range(self.range_outliers):
                        d = now - datetime.timedelta(seconds=records - i - j)
                        value = self.generate_outliers(multiplier, j)
                        file.write(f'{value:.2f},{d.strftime("%Y-%m-%dT%H:%M:%SZ")}\n')
                        try:
                            next(iter_range)
                        except StopIteration:
                            return None
                else:
                    d = now - datetime.timedelta(seconds=records - i)
                    value = random.uniform(self.start, self.end)
                    file.write(f'{value:.2f},{d.strftime("%Y-%m-%dT%H:%M:%SZ")}\n')

    def generate_outliers(self, multiplier, value):
        # 10000 = (range/2)**2
        a = -1*multiplier/((self.range_outliers/2)**2)
        b = -self.range_outliers * a
        y = a*(value**2) + b*value + self.start
        return random.uniform(y - 0.4, y + 0.4)


    def generate_without_outliers(self, records, filename):
        now = datetime.datetime.now()
        with open(filename, 'w') as file:
            for i in range(records):
                d = now - datetime.timedelta(seconds=records - i)
                value = random.uniform(self.start, self.end)
                file.write(f'{value:.2f},{d.strftime("%Y-%m-%dT%H:%M:%SZ")}\n')

    def plot_dataset(self, filename):
        data = pd.read_csv(filename, names=['value', 'datetime'])
        data = pd.DataFrame(data)
        data['datetime'] = pd.to_datetime(data['datetime'])
        figsize = (25, 3)
        data.plot(x='datetime', y='value', figsize=figsize)
        # plt.grid()
        plt.show()


if __name__ == '__main__':
    # Temperature 30.00-31.00
    # Humidity 28.00-29.00
    g = Generator(30, 31)
    # g.generate_with_outliers(1000, 'train_dataset_outliers.csv')
    # g.generate_with_outliers(5000, 'test_dataset_outliers.csv')
    #
    # g.generate_without_outliers(1000, 'train_dataset.csv')
    # g.generate_without_outliers(5000, 'test_dataset.csv')

    # g.plot_dataset('train_dataset_outliers.csv')
    g.plot_dataset('test_dataset_outliers.csv')
    # g.plot_dataset('train_dataset.csv')
    # g.plot_dataset('test_dataset.csv')
