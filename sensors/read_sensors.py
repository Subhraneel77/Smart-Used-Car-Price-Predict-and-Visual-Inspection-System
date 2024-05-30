import RPi.GPIO as GPIO
import time

# Setup GPIO pins
GPIO.setmode(GPIO.BCM)
mileage_pin = 18
engine_health_pin = 23

GPIO.setup(mileage_pin, GPIO.IN)
GPIO.setup(engine_health_pin, GPIO.IN)

def read_sensors():
    mileage = GPIO.input(mileage_pin)
    engine_health = GPIO.input(engine_health_pin)
    return mileage, engine_health

if __name__ == "__main__":
    try:
        while True:
            mileage, engine_health = read_sensors()
            print(f"Mileage: {mileage}, Engine Health: {engine_health}")
            time.sleep(1)
    except KeyboardInterrupt:
        GPIO.cleanup()
