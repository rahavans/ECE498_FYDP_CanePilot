from gpiozero import PWMOutputDevice
from time import sleep
import threading

from queue import Queue, Empty
import threading, time
from gpiozero import PWMOutputDevice

class MotorController:
    def __init__(self, motor_pins, pwm_freq=200):
        """
        motor_pins: list of BCM pin numbers
        pwm_freq  : Hz; 200 Hz gives you 8 PWM cycles in a 40 ms pulse
        """
        self.motors  = [PWMOutputDevice(pin, frequency=pwm_freq, initial_value=0)
                        for pin in motor_pins]
        self.queues  = [Queue(maxsize=10) for _ in motor_pins]   # small back-pressure
        self.stop_evt = threading.Event()

        # start worker threads
        self.workers = []
        for motor, q in zip(self.motors, self.queues):
            t = threading.Thread(target=self._worker,
                                 args=(motor, q),
                                 daemon=True)
            t.start()
            self.workers.append(t)

    # ---------------- internal ----------------
    def _worker(self, motor, q: Queue):
        while not self.stop_evt.is_set():
            try:
                power, duration_ms = q.get(timeout=0.1)
            except Empty:
                continue

            if power > 0 and duration_ms > 0:
                motor.value = power          # start PWM
                time.sleep(duration_ms / 1000.0)
            motor.off()
            q.task_done()

    # ---------------- public API -------------
    def vibrate_motors(self, config):
        """
        config: list/tuple like [(power, ms), ...] matching self.motors order
        """
        if len(config) != len(self.motors):
            raise ValueError("Config length must match number of motors")

        for (power, dur), q in zip(config, self.queues):
            # overwrite any stale command so the newest wins
            with q.mutex:
                q.queue.clear()
            q.put_nowait((power, dur))

    def stop_all(self):
        for m in self.motors:
            m.off()

    def close(self):
        self.stop_evt.set()
        # unblock workers
        for q in self.queues:
            q.put_nowait((0, 0))
        for t in self.workers:
            t.join()
        self.stop_all()

# -------------------------------
# Test case
# -------------------------------
if __name__ == "__main__":
    motor_pins = [23, 17, 25]
    controller = MotorController(motor_pins)

    try:
        print("Running test vibration pattern...")
        controller.vibrate_motors([
            (1.0, 1200000),  # (power, ms)
            (1.0, 1200000),
            (1.0, 1200000)
        ])
        print("Test complete.")
        sleep(4)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        controller.close()
        print("All motors off. Controller closed.")
