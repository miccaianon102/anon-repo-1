import sys


class DualLogger:
    def __init__(self, filepath: str, mode: str = 'a'):
        self.terminal = sys.stdout
        self.log = open(filepath, mode, encoding='utf-8')

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


class GradualSTARScheduler:
    """Linear ramp-up for the STAR loss weight from 0 → 1 over [start, end] epochs."""

    def __init__(self, start_epoch: int = 10, end_epoch: int = 20):
        self.start_epoch  = start_epoch
        self.end_epoch    = end_epoch
        self.ramp_length  = max(end_epoch - start_epoch, 1)

    def get_weight(self, epoch: int) -> float:
        if epoch < self.start_epoch:
            return 0.0
        if epoch >= self.end_epoch:
            return 1.0
        return (epoch - self.start_epoch) / self.ramp_length
