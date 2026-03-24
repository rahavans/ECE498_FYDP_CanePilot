from gpiozero import PWMOutputDevice
from time import sleep

# Define GPIO pins
motor_pins = [23, 17, 25]

# Create PWMOutputDevice instances for each pin
motors = [PWMOutputDevice(pin, frequency=150) for pin in motor_pins]

try:
    # Set PWM duty cycle to 100% (full speed)
    for motor in motors:
        motor.value = 1.0  # 1.0 = 100% duty cycle

    print("Motors running at ~9000 RPM (150Hz)")
    while True:
        sleep(1)  # Keep running

except KeyboardInterrupt:
    print("Stopping motors...")
    for motor in motors:
        motor.off()
