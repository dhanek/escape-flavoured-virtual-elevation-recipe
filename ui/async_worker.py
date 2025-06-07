from PySide6.QtCore import QObject, Signal, Slot, QMutex, QMutexLocker
import copy

class AsyncWorker(QObject):
    trigger = Signal()
    resultReady = Signal(object)

    def __init__(self):
        super().__init__()
        self._mutex = QMutex()
        self._working = False
        self._new_values = False
        self._latest_values = {}
        self.trigger.connect(self.process)

    def set_values(self, values: dict):
        with QMutexLocker(self._mutex):
            self._latest_values = copy.deepcopy(values)
            self._new_values = True
            if self._working:
                return
            self._working = True
        self.trigger.emit()

    def get_values(self):
        return copy.deepcopy(self._latest_values)

    @Slot()
    def process(self):
        self._mutex.lock()

        while self._new_values:
            latest_values = self._latest_values.copy()
            self._new_values = False

            self._mutex.unlock()
            result = self._process_value(latest_values)
            result = copy.deepcopy(result)
            self.resultReady.emit(result)
            self._mutex.lock()

        self._working = False
        self._mutex.unlock()

    def _process_value(self, value: dict):
        """
        Override this method in subclass.
        """
        raise NotImplementedError("Subclasses must implement _process_value")
