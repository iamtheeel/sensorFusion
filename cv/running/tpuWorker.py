import multiprocessing
import traceback
import numpy as np

def tpu_worker_loop(model_path, input_queue, output_queue):
    """Subprocess loop that owns the TPU interpreter and handles inference."""
    try:
        print("[TPUWorker] Subprocess started")
        import pycoral.utils.edgetpu as etpu

        interpreter = etpu.make_interpreter(model_path)
        print("[TPUWorker] Interpreter created")
        interpreter.allocate_tensors()
        print("[TPUWorker] Tensors allocated")

        in_details = interpreter.get_input_details()
        out_details = interpreter.get_output_details()
        input_index = in_details[0]['index']
        output_index = out_details[0]['index']

        while True:
            data = input_queue.get()
            if data == "STOP":
                print("[TPUWorker] Stopping subprocess")
                break
            try:
                interpreter.set_tensor(input_index, data)
                interpreter.invoke()
                result = interpreter.get_tensor(output_index).copy()
                output_queue.put(("ok", result))
            except Exception as e:
                output_queue.put(("error", str(e)))

    except Exception:
        msg = traceback.format_exc()
        print("[TPUWorker] Fatal startup error:\n", msg)
        try:
            output_queue.put(("fatal", msg))
        except:
            pass

class TPUWorker:
    def __init__(self, model_path, timeout=1.0):
        self.model_path = model_path
        self.timeout = timeout
        self.process = None
        self.input_queue = None
        self.output_queue = None
        self._start_worker()

    def _start_worker(self):
        ctx = multiprocessing.get_context("spawn")
        self.input_queue = ctx.Queue()
        self.output_queue = ctx.Queue()
        self.process = ctx.Process(
            target=tpu_worker_loop,
            args=(self.model_path, self.input_queue, self.output_queue)
        )
        self.process.start()

    def infer(self, input_data):
        if not self.process or not self.process.is_alive():
            return "dead", None

        try:
            self.input_queue.put(input_data)
            status, result = self.output_queue.get(timeout=self.timeout)
            return status, result
        except multiprocessing.queues.Empty:
            return "timeout", None
        except (BrokenPipeError, EOFError):
            return "dead", None

    def restart(self):
        self.stop()
        self._start_worker()

    def stop(self):
        if self.process and self.process.is_alive():
            try:
                self.input_queue.put("STOP")
                self.process.join(timeout=1)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join()
            except Exception:
                pass
        if self.process:
            self.process.close()
        if self.input_queue:
            self.input_queue.close()
        if self.output_queue:
            self.output_queue.close()

    def __del__(self):
        self.stop()
