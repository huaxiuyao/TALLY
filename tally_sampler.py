class TallySampler:
    def __init__(self, dataset, batch_size, iter_num) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.iter_num = iter_num

    def __iter__(self):
        for i in range(self.iter_num):
            yield from self.dataset.get_balance_pair(self.batch_size)

    def __len__(self) -> int:
        return self.iter_num * 2 * self.batch_size
