import dbus
import time
import subprocess
import threading
import logging
import queue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global queue and worker thread for speech
speech_queue = queue.Queue()
speech_thread_started = threading.Event()

def _speech_worker():
    while True:
        text, voice, speed = speech_queue.get()
        try:
            logger.info(f"Speaking: {text}")
            subprocess.run(
                ["espeak", "-v", voice, text, "-s", str(speed)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            logger.error(f"Failed to speak text: {e}")
        speech_queue.task_done()

def speak_text(text, voice="mb-en1", speed=150):
    """
    Queued, non-overlapping function to speak text using espeak
    
    Args:
        text (str): The text to speak
        voice (str): Voice to use (default: mb-en1)
        speed (int): Speech rate (default: 150)
    """
    if not speech_thread_started.is_set():
        thread = threading.Thread(target=_speech_worker, daemon=True)
        thread.start()
        speech_thread_started.set()

    speech_queue.put((text, voice, speed))

# init and set_default_sink unchanged...


def init(mac_address):
    # Connect to system bus
    bus = dbus.SystemBus()
    adapter_path = "/org/bluez/hci0"  # Typically hci0 is your Bluetooth adapter
    adapter = dbus.Interface(bus.get_object("org.bluez", adapter_path), "org.bluez.Adapter1")

    # Power on the adapter
    adapter_props = dbus.Interface(bus.get_object("org.bluez", adapter_path), "org.freedesktop.DBus.Properties")
    adapter_props.Set("org.bluez.Adapter1", "Powered", dbus.Boolean(1))
    logger.info("Bluetooth powered on.")

    # Start discovery to ensure the device is visible (optional)
    adapter.StartDiscovery()
    logger.info("Discovering devices...")
    time.sleep(5)
    adapter.StopDiscovery()

    device_path = f"/org/bluez/hci0/dev_{mac_address.replace(':', '_')}"
    try:
        device = dbus.Interface(bus.get_object("org.bluez", device_path), "org.bluez.Device1")
    except dbus.exceptions.DBusException:
        logger.error(f"Device {mac_address} not found. Try ensuring it's in pairing mode.")
        return

    # Get device properties to check paired status
    device_props = dbus.Interface(device, "org.freedesktop.DBus.Properties")
    is_paired = device_props.Get("org.bluez.Device1", "Paired")

    if is_paired:
        logger.info(f"Device {mac_address} is already paired")
        return

    # Pair the device if not already paired
    logger.info(f"Pairing with {mac_address}...")
    try:
        device.Pair()
        logger.info(f"Paired with {mac_address}")
    except dbus.exceptions.DBusException as e:
        logger.error(f"Pairing failed: {e}")

    set_default_sink(mac_address)

def set_default_sink(mac_address):
    # Convert MAC to expected sink name format (e.g., bluez_output.94_DB_56_F0_6C_30)
    sink_prefix = "bluez_output." + mac_address.replace(":", "_")

    # List sinks and find matching one
    output = subprocess.check_output(["pactl", "list", "short", "sinks"]).decode()
    for line in output.splitlines():
        if sink_prefix in line:
            sink_name = line.split("\t")[1]
            logger.info(f"Found sink: {sink_name}")
            subprocess.run(["pactl", "set-default-sink", sink_name])
            logger.info("Default sink set to Bluetooth device.")
            return

    logger.error("Sink not found for device.")

if __name__ == "__main__":
    target_mac = "94:DB:56:F0:6C:30"  # Replace with your device address
    init(target_mac)
