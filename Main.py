
import cv2
import serial
from ultralytics import YOLO

# -------------------- Configuration --------------------

MODEL_PATH = "yolo11n.pt"
SERIAL_PORT = "COM6"
BAUD_RATE = 115200

CENTER_TOLERANCE = 15

# -------------------- Initialization --------------------

model = YOLO(MODEL_PATH)
model.to("cuda:0")

cap = cv2.VideoCapture(0)

arduino = serial.Serial(
    SERIAL_PORT,
    BAUD_RATE,
    timeout=0
)

last_distance = None
serial_buffer = bytearray()


# -------------------- Distance Sensor --------------------

def get_distance():
    global last_distance

    # Read available bytes without blocking
    waiting = arduino.in_waiting

    if waiting > 0:
        serial_buffer.extend(arduino.read(waiting))

    # Process complete lines only
    while b"\n" in serial_buffer:
        line, _, remaining = serial_buffer.partition(b"\n")
        serial_buffer[:] = remaining

        data = line.decode(
            "utf-8",
            errors="replace"
        ).strip()

        if data.startswith("Distance:"):
            try:
                distance = int(data.split()[1])

                if distance >= 0:
                    last_distance = distance

            except (ValueError, IndexError):
                pass

        elif data == "Invalid measurement":
            last_distance = None

    return last_distance


# -------------------- Object Position --------------------

def is_centered(results, height, width):
    if len(results[0].boxes) == 0:
        return None

    box = results[0].boxes[0]

    x1, y1, x2, y2 = map(int, box.xyxy[0])

    objectX = (x1 + x2) / 2
    objectY = (y1 + y2) / 2

    errorX = objectX - width / 2
    errorY = objectY - height / 2

    isCenter = (
        abs(errorX) <= CENTER_TOLERANCE
        and abs(errorY) <= CENTER_TOLERANCE
    )

    return errorX, errorY, isCenter


# -------------------- Drawing Functions --------------------

def draw_center(frame, is_center):
    height, width = frame.shape[:2]

    center = (
        width // 2,
        height // 2
    )

    if is_center:
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)

    cv2.circle(
        frame,
        center,
        10,
        color,
        -1
    )


def draw_distance(frame, distance_mm):
    if distance_mm is None:
        text = "Distance: N/A"
    else:
        text = f"Distance: {distance_mm} mm"

    cv2.putText(
        frame,
        text,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


def draw_errors(frame, errorX, errorY):
    if errorX is None or errorY is None:
        text = "Object: Not detected"
    else:
        text = f"X: {errorX:.0f}, Y: {errorY:.0f}"

    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


# -------------------- Main Loop --------------------

def detect_object():
    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Camera error")
                break

            # Rotate camera image by 180 degrees
            frame = cv2.rotate(
                frame,
                cv2.ROTATE_180
            )

            height, width = frame.shape[:2]

            # Detect bottles only (COCO class 39)
            results = model(
                frame,
                classes=[39],
                max_det=1,
                conf=0.5,
                verbose=False
            )

            # Calculate object position
            result = is_centered(
                results,
                height,
                width
            )

            if result is not None:
                errorX, errorY, isCenter = result
            else:
                errorX = None
                errorY = None
                isCenter = False

            # Read distance sensor
            distance_mm = get_distance()

            # Draw YOLO detections
            annotated_frame = results[0].plot()

            # Draw additional information
            draw_center(annotated_frame, isCenter)

            draw_errors(
                annotated_frame,
                errorX,
                errorY
            )

            draw_distance(
                annotated_frame,
                distance_mm
            )

            # Display frame
            cv2.imshow(
                "Robot Arm - Object Detection",
                annotated_frame
            )

            # Press Q to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        arduino.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_object()
