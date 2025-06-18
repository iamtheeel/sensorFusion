import multiprocessing
import traceback
import numpy as np
import time
from pycoral.adapters import common

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tpuWorker")


def tpu_worker_loop(model_path, input_queue, output_queue):
    try:
        print("[TPUWorker] Subprocess started")
        import numpy as np
        from tflite_runtime.interpreter import Interpreter, load_delegate
        from pycoral.utils.edgetpu import load_edgetpu_delegate

        try:
            time.sleep(0.1)
            delegate = load_edgetpu_delegate({'device': 'pci'})  # or 'pci' if you're using PCIe
            print("[TPUWorker] Delegate loaded")
        except Exception as e:
            print("[TPUWorker] Failed to load delegate:", e)
            output_queue.put(("fatal", f"Failed to load Edge TPU delegate: {e}"))
            return

        interpreter = Interpreter(model_path=model_path, experimental_delegates=[delegate])
        print("[TPUWorker] Interpreter created")

        interpreter.allocate_tensors()
        print("[TPUWorker] Tensors allocated")

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        #print(f"input_details: {input_details}")
        #print(f"output_details: {output_details}")

        #Set the zero and scale for the input matrix (x)
        input_zero = input_details[0]['quantization'][1]
        input_scale = input_details[0]['quantization'][0]
        output_zero = output_details[0]['quantization'][1]
        output_scale = output_details[0]['quantization'][0]
        # If the model isn't quantized then these should be zero
        # Check against small epsilon to avoid comparing float/int
        if input_scale < 1e-9: input_scale = 1.0
        if output_scale < 1e-9: output_scale = 1.0

        #print("Input scale: {}".format(input_scale))
        #print("Input zero: {}".format(input_zero))
        #print("Output scale: {}".format(output_scale))
        #print("Output zero: {}".format(output_zero))
        print("Successfully loaded {}".format(model_path))

        ## get_image_size
        input_size = common.input_size(interpreter)
        print("input_size: {}".format(input_size))


        ## Set the in and out index for the queue
        input_index = input_details[0]['index']
        output_index = output_details[0]['index']

        output_queue.put({"status": "ready",
                          "input_zero": input_zero,
                          "input_scale": input_scale,
                          "output_zero": output_zero,
                          "output_scale": output_scale,
                          "input_size": input_size
                        })

        while True:
            data = input_queue.get()
            if isinstance(data, str) and data == "STOP":
                break
            try:
                #print("Received input:", type(data), getattr(data, "shape", None), getattr(data, "dtype", None))
                interpreter.set_tensor(input_index, data)
                interpreter.invoke()
                result = interpreter.get_tensor(output_index).copy()
                output_queue.put(("ok", result))
            except Exception as e:
                output_queue.put(("error", str(e)))

    except Exception as e:
        import traceback
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
        ctx = multiprocessing.get_context("fork")
        self.input_queue = ctx.Queue()
        self.output_queue = ctx.Queue()
        self.process = ctx.Process(
            target=tpu_worker_loop,
            args=(self.model_path, self.input_queue, self.output_queue)
        )
        self.process.start()

        # Wait for the init data
        try:
            init_info = self.output_queue.get(timeout=2.0) # Get the init information
            if init_info.get("status") == "ready":
                self.input_zero = init_info.get("input_zero") 
                self.input_scale = init_info.get("input_scale") 
                self.output_zero = init_info.get("output_zero") 
                self.output_scale = init_info.get("output_scale") 
                self.input_size = init_info.get("input_size") 
            else:
                raise RuntimeError(f"TPU worker failed on start: {str(info)}")
        except Exception as e:
            raise RuntimeError(f"TPU worker Did Not Return MetaData")


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
