from gpiozero import Button
from threading import Timer, Thread
import time

BUTTON_PIN = 16
HOLD_TIME = 3  # seconds

# Callback references
_single_click_callback = None
_single_click_args = ()

_double_click_callback = None
_double_click_args = ()

_triple_click_callback = None
_triple_click_args = ()

_hold_callback = None
_hold_args = ()

_hold_release_callback = None
_hold_release_args = ()

# Add global state to track release
_hold_release_fired = False

# Internal state
_clicks = 0
_last_press_time = 0
_hold_timer = None
_hold_detected = False

button = Button(BUTTON_PIN, pull_up=True, bounce_time=0.05)

def _handle_pressed():
    global _last_press_time, _clicks, _hold_timer, _hold_detected, _hold_release_fired
    _last_press_time = time.time()
    _clicks += 1
    _hold_detected = False
    _hold_release_fired = False
    _hold_timer = Timer(HOLD_TIME, _check_hold)
    _hold_timer.start()

def _handle_released():
    global _clicks, _hold_timer, _hold_detected, _hold_release_fired

    if _hold_timer:
        _hold_timer.cancel()

    if _hold_detected and not _hold_release_fired:
        _hold_release_fired = True
        if _hold_release_callback:
            _hold_release_callback(*_hold_release_args)
        _clicks = 0
        return

    # Start thread to check click count after timeout
    Thread(target=_check_clicks).start()

def _check_clicks():
    global _clicks
    time.sleep(0.35)  # Wait to see if more clicks follow
    if _clicks == 1 and _single_click_callback:
        _single_click_callback(*_single_click_args)
    elif _clicks == 2 and _double_click_callback:
        _double_click_callback(*_double_click_args)
    elif _clicks >= 3 and _triple_click_callback:
        _triple_click_callback(*_triple_click_args)
    _clicks = 0

def _check_hold():
    global _clicks, _hold_detected
    if button.is_pressed:
        _hold_detected = True
        _clicks = 0
        if _hold_callback:
            _hold_callback(*_hold_args)

# Registration functions
def register_single_click(callback, args=()):
    global _single_click_callback, _single_click_args
    _single_click_callback = callback
    _single_click_args = args

def register_double_click(callback, args=()):
    global _double_click_callback, _double_click_args
    _double_click_callback = callback
    _double_click_args = args

def register_triple_click(callback, args=()):
    global _triple_click_callback, _triple_click_args
    _triple_click_callback = callback
    _triple_click_args = args

def register_hold(callback, args=()):
    global _hold_callback, _hold_args
    _hold_callback = callback
    _hold_args = args

def register_hold_release(callback, args=()):
    global _hold_release_callback, _hold_release_args
    _hold_release_callback = callback
    _hold_release_args = args

def setup_button(hold_time=HOLD_TIME):
    global HOLD_TIME
    HOLD_TIME = hold_time
    button.when_pressed = _handle_pressed
    button.when_released = _handle_released

# Example usage
if __name__ == "__main__":
    def on_single(name):
        print(f"Single click from {name}")

    def on_double(name, level):
        print(f"Double click from {name} (level {level})")

    def on_triple(name):
        print(f"Triple click from {name}")

    def on_hold():
        print("Held for 3+ seconds!")

    def on_hold_release():
        print("Button was released after a 3+ second hold!")

    setup_button(hold_time=3)
    register_single_click(on_single, args=("Button1",))
    register_double_click(on_double, args=("Button1", 5))
    register_triple_click(on_triple, args=("Button1",))
    register_hold(on_hold)
    register_hold_release(on_hold_release)

    print("Button ready. Press Ctrl+C to quit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting cleanly.")
