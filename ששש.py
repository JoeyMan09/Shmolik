
import serial

arduino = serial.Serial("COM6", 115200, timeout=1)

print("Connected to COM6")

try:
    while True:
        data = arduino.readline().decode(
            "utf-8", errors="replace"
        ).strip()

        if data:
            print(data)

except KeyboardInterrupt:
    print("Disconnected")

finally:
    arduino.close()
