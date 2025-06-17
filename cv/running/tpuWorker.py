import multiprocessing
import time
import traceback
import numpy as np

class TPUWorker:
    def __init__(self, model_path, timeout=1.0):
        self.model_path = model_path
        self.timeout = timeout
        self._start_worker()

    def _start_worker(self):
        ctx = multiprocessing.get_context("spawn")
        self.input_queue = ctx.Queue()
        self.output_queue = ctx.Queue()
        self.process = ctx.Process(target=self._worker_loop, args=(
            self.model_path, self.input_queue, self.output_queue))
        self.process.start()

    def _worker_loop(self, model_path, input_queue, output_queue):
        try:
            import pycoral.utils.edgetpu as etpu
            from pycoral.adapters.common import input_details, output_details
            import numpy as np

            interpreter = etpu.make_interpreter(model_path)
            interpreter.allocate_tensors()

            input_index = input_details(interpreter)[0]['index']
            output_index = output_details(interpreter)[0]['index']

            while True:
                data = input_queue.get()
                if data == "STOP":
                    break
                try:
                    interpreter.set_tensor(input_index, data)
                    interpreter.invoke()
                    result = interpreter.get_tensor(output_index).copy()
                    output_queue.put(("ok", result))
                except Exception as e:
                    output_queue.put(("error", str(e)))
        except Exception as e:
            output_queue.put(("fatal", traceback.format_exc()))

    def infer(self, input_data):
        if not self.process.is_alive():
            return "dead", None

        self.input_queue.put(input_data)
        try:
            status, result = self.output_queue.get(timeout=self.timeout)
            return status, result
        except multiprocessing.queues.Empty:
            return "timeout", None

    def restart(self):
        self.stop()
        self._start_worker()

    def stop(self):
        if self.process.is_alive():
            self.input_queue.put("STOP")
            self.process.join(timeout=1)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join()
        self.process.close()
        self.input_queue.close()
        self.output_queue.close()

    def __del__(self):
        self.stop()
