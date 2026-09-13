import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import threading
import time


# ============================================================
# SETTINGS
# ============================================================

CAMERA_INDEX = 0

# Your laptop microphone
MICROPHONE_DEVICE = 1

SAMPLE_RATE = 44100
CHANNELS = 1
BLOCK_SIZE = 1024

# Clap sensitivity
CLAP_THRESHOLD = 0.004
PEAK_THRESHOLD = 0.015

# Prevent one clap from triggering multiple times
CLAP_COOLDOWN = 1.5


# ============================================================
# GLOBAL VARIABLES
# ============================================================

clap_detected = False
last_clap_time = 0

lock = threading.Lock()

last_print_time = 0


# ============================================================
# MICROPHONE CALLBACK
# ============================================================

def audio_callback(indata, frames, time_info, status):

    global clap_detected
    global last_clap_time
    global last_print_time

    if status:
        print("Audio:", status)

    # Get microphone samples
    audio = indata[:, 0]

    # Calculate RMS volume
    volume = float(np.sqrt(np.mean(audio ** 2)))

    # Calculate loudest point
    peak = float(np.max(np.abs(audio)))

    current_time = time.time()

    # Print microphone level every 0.5 seconds
    if current_time - last_print_time > 0.5:

        print(
            f"Mic volume: {volume:.4f} | Peak: {peak:.4f}"
        )

        last_print_time = current_time

    # --------------------------------------------------------
    # CLAP DETECTION
    # --------------------------------------------------------

    if (
        volume > CLAP_THRESHOLD
        and peak > PEAK_THRESHOLD
        and current_time - last_clap_time > CLAP_COOLDOWN
    ):

        with lock:
            clap_detected = True

        last_clap_time = current_time

        print()
        print("👏👏👏 CLAP DETECTED! 👏👏👏")
        print()


# ============================================================
# START MICROPHONE
# ============================================================

print()
print("==============================================")
print("STARTING MICROPHONE")
print("==============================================")

try:

    microphone = sd.InputStream(
        device=MICROPHONE_DEVICE,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        blocksize=BLOCK_SIZE,
        dtype="float32",
        callback=audio_callback
    )

    microphone.start()

    print("✅ Microphone started!")
    print("🎤 Using: Microphone Array (Senary Audio)")
    print()

except Exception as e:

    print("❌ Microphone error:")
    print(e)
    exit()


# ============================================================
# START CAMERA
# ============================================================

print("==============================================")
print("STARTING CAMERA")
print("==============================================")

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():

    print("❌ Camera could not be opened.")

    microphone.stop()
    microphone.close()

    exit()

print("✅ Camera opened!")
print()


# ============================================================
# CAPTURE BACKGROUND
# ============================================================

print("==============================================")
print("BACKGROUND CAPTURE")
print("==============================================")
print()
print("Move away from the camera.")
print("Capturing background...")
print()

background = None

for i in range(100):

    ret, frame = cap.read()

    if not ret:

        print("❌ Camera frame error.")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # Display
    cv2.imshow(
        "Preparing Background - Move Away",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        cap.release()

        microphone.stop()
        microphone.close()

        cv2.destroyAllWindows()

        exit()


# Save background

if ret:

    background = frame.copy()

else:

    print("❌ Background capture failed.")

    cap.release()

    microphone.stop()
    microphone.close()

    cv2.destroyAllWindows()

    exit()


cv2.destroyWindow(
    "Preparing Background - Move Away"
)

print()
print("✅ Background captured!")
print()


# ============================================================
# MEDIAPIPE
# ============================================================

mp_selfie = mp.solutions.selfie_segmentation


# ============================================================
# INITIAL STATE
# ============================================================

invisible = False


# ============================================================
# INSTRUCTIONS
# ============================================================

print("==============================================")
print("AI-POWERED CLAP INVISIBILITY SYSTEM")
print("==============================================")
print()
print("👏 Clap once  → INVISIBLE")
print("👏 Clap again → VISIBLE")
print()
print("Q → EXIT")
print()
print("🎤 Listening for claps...")
print("==============================================")
print()


# ============================================================
# SELFIE SEGMENTATION
# ============================================================

with mp_selfie.SelfieSegmentation(
    model_selection=1
) as segment:

    while True:

        # ----------------------------------------------------
        # READ CAMERA
        # ----------------------------------------------------

        ret, frame = cap.read()

        if not ret:

            print("❌ Could not read camera frame.")
            break


        # Mirror camera
        frame = cv2.flip(frame, 1)


        # ----------------------------------------------------
        # RESIZE BACKGROUND
        # ----------------------------------------------------

        background_resized = cv2.resize(
            background,
            (frame.shape[1], frame.shape[0])
        )


        # ----------------------------------------------------
        # CHECK CLAP
        # ----------------------------------------------------

        with lock:

            if clap_detected:

                # Toggle
                invisible = not invisible

                # Reset signal
                clap_detected = False

                if invisible:

                    print("🪄 STATUS CHANGED → INVISIBLE")

                else:

                    print("👤 STATUS CHANGED → VISIBLE")


        # ====================================================
        # VISIBLE MODE
        # ====================================================

        if not invisible:

            output = frame.copy()


        # ====================================================
        # INVISIBLE MODE
        # ====================================================

        else:

            # Convert BGR → RGB
            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # AI segmentation
            result = segment.process(rgb)

            # Get segmentation mask
            mask = result.segmentation_mask

            # Smooth mask
            mask = cv2.GaussianBlur(
                mask,
                (7, 7),
                0
            )

            # Person area
            condition = mask > 0.5

            # Start with camera
            output = frame.copy()

            # Replace person with background
            output[condition] = background_resized[condition]


        # ====================================================
        # STATUS
        # ====================================================

        if invisible:

            status = "INVISIBLE"

        else:

            status = "VISIBLE"


        # ====================================================
        # STATUS BOX
        # ====================================================

        cv2.rectangle(
            output,
            (10, 10),
            (410, 80),
            (0, 0, 0),
            -1
        )


        # ====================================================
        # STATUS TEXT
        # ====================================================

        cv2.putText(
            output,
            f"STATUS: {status}",
            (25, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )


        # ====================================================
        # INSTRUCTIONS
        # ====================================================

        cv2.putText(
            output,
            "CLAP = TOGGLE | Q = EXIT",
            (20, output.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow(
            "AI Invisibility System",
            output
        )


        # ====================================================
        # KEYBOARD
        # ====================================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            print()
            print("Exiting...")
            break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

microphone.stop()
microphone.close()

cv2.destroyAllWindows()

print()
print("✅ AI Invisibility System stopped.")