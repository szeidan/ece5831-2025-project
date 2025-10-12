# Rock-Paper-Scissors with Teachable Machine and Live Camera
# ----------------------------------------------------------
# This version includes:
# - Robust label parsing (handles "0 Rock" or "Rock")
# - Canonicalization so 'scissors1'/'scissors2' → 'scissors'
# - Computer chooses by LABEL (not index) to avoid IndexError
# - Safer key handling (p/P/SPACE to capture, q/Q/ESC to quit)
# - Window brought to front to avoid lost key events on Windows
# - Helpful comments explaining why each change was made

from tensorflow.keras.models import load_model  # use TF's Keras (no standalone keras needed)
from PIL import Image, ImageOps
import numpy as np
import cv2
import random
from typing import List, Tuple


class RockPaperScissors:
    def __init__(self, model_path: str = 'model/keras_model.h5', labels_path: str = 'model/labels.txt'):
        # Load the trained model
        self.model = load_model(model_path, compile=False)

        # --- Robust label parsing ---
        # WHY: Teachable Machine labels can be lines like "0 Rock" or just "Rock".
        # We normalize everything to lowercase text: ['rock','paper','scissors', ...]
        with open(labels_path, 'r', encoding='utf-8') as f:
            raw = [ln.strip() for ln in f if ln.strip()]

        self.class_names: List[str] = []
        for ln in raw:
            parts = ln.split(maxsplit=1)
            # if line starts with an integer index, take the text after it
            label = parts[1] if (len(parts) > 1 and parts[0].isdigit()) else ln
            self.class_names.append(label.strip().lower())

        # Image size expected by Teachable Machine (most export at 224x224)
        self.size: Tuple[int, int] = (224, 224)

        # --- Canonical emoji map ---
        # WHY: We canonicalize any 'scissors1'/'scissors2' → 'scissors', so one key is enough.
        self.emoji = {
            "rock": "✊",
            "paper": "✋",
            "scissors": "✌️"
        }

    # ------------------------- helpers -------------------------

    def _canonical(self, label_or_idx):
        """
        Normalize any input to 'rock'/'paper'/'scissors'.

        Accepts:
          - int class index
          - string label (e.g., 'rock', 'scissors1')
          - digit-string (e.g., '0', '1', '2') → mapped via class_names

        WHY: Teachable Machine sometimes emits 'scissors1/2'. We squash them to 'scissors'
        so our game logic and emoji map stay simple and bug-free.
        """
        if isinstance(label_or_idx, int):
            # index → label (with bounds check)
            if 0 <= label_or_idx < len(self.class_names):
                label = self.class_names[label_or_idx]
            else:
                return "scissors"  # safe fallback
        else:
            s = str(label_or_idx).strip().lower()
            if s.isdigit():  # digit-string → treat like index
                i = int(s)
                if 0 <= i < len(self.class_names):
                    label = self.class_names[i]
                else:
                    label = s  # unexpected, but won't match wins keys
            else:
                label = s

        if label.startswith("scissor"):  # covers 'scissors', 'scissors1', 'scissors2', etc.
            return "scissors"
        return label

    def _prep_frame(self, frame_bgr):
        """
        Resize/center-crop to model size and normalize to [-1, 1], as Teachable Machine expects.
        Using PIL ImageOps.fit ensures we always feed exactly 224x224 without distortion.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil = ImageOps.fit(pil, self.size, Image.Resampling.LANCZOS)
        arr = np.asarray(pil).astype(np.float32)
        arr = (arr / 127.5) - 1.0
        return np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)

    def _predict(self, frame_bgr):
        """
        Return (index, confidence) for a live frame.
        """
        batch = self._prep_frame(frame_bgr)
        preds = self.model.predict(batch, verbose=0)
        idx = int(np.argmax(preds))
        conf = float(preds[0][idx])
        return idx, conf

    # Public helper for notebook/tests
    def predict_image(self, image_path: str):
        """
        Predict a static image file, returning (label, confidence).
        """
        pil = Image.open(image_path).convert("RGB")
        pil = ImageOps.fit(pil, self.size, Image.Resampling.LANCZOS)
        arr = np.asarray(pil).astype(np.float32)
        arr = (arr / 127.5) - 1.0
        batch = np.expand_dims(arr, axis=0)
        preds = self.model.predict(batch, verbose=0)
        idx = int(np.argmax(preds))
        return self._canonical(idx), float(preds[0][idx])

    def _computer_choice(self):
        """
        Return a LABEL, not an index.
        WHY: If labels.txt has only 3 classes, returning an index in [0..3] can crash (IndexError).
        Returning a canonical label avoids any dependency on label list length.
        """
        return random.choice(["rock", "paper", "scissors"])

    def _get_winner(self, computer_choice, user_choice):
        """
        Return 'tie' | 'computer' | 'user' using canonical labels.

        WHY: Comparing canonical labels makes logic simple and immune to label variations like
        'scissors1'/'scissors2'.
        """
        uc = self._canonical(user_choice)
        cc = self._canonical(computer_choice)
        if uc == cc:
            return "tie"
        wins = {"rock": "scissors", "paper": "rock", "scissors": "paper"}
        return "user" if wins.get(uc) == cc else "computer"

    # --------------------------- game ---------------------------

    def play(self, camera_index: int = 0):
        """
        Live webcam demo:
          - Click the video window to give it focus (so keys go to OpenCV).
          - Press 'p' (or SPACE) to capture and evaluate your hand.
          - Press 'q' (or ESC) to quit.

        WHY these UI tweaks:
         - Named window + TOPMOST: helps ensure key events reach the OpenCV window on Windows.
         - Multiple keys: more forgiving in demos (p/P/SPACE, q/Q/ESC).
        """
        win_name = "Rock-Paper-Scissors"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        try:
            cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1)
        except cv2.error:
            pass  # some builds don't support TOPMOST, safe to ignore

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("Error: Could not open video stream. Try camera_index=1 or 2.")
            return

        # Friendly welcome line (now uses 'scissors' only; no 'scissors1' KeyError)
        print("=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*")
        print(
            f"👾 Welcome to {self.emoji['rock']}, {self.emoji['paper']}, {self.emoji['scissors']}   Rock-Paper-Scissors!"
        )
        print("📌 Click the camera window to focus it.")
        print("   Press 'p' (or SPACE) to capture a frame, 'q' (or ESC) to quit.")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Failed to grab frame.")
                    break

                cv2.imshow(win_name, frame)
                key = cv2.waitKey(15) & 0xFF  # 15ms tick; small but gives time for key events

                if key in (ord('p'), ord('P'), 32):  # 32 = SPACE
                    idx, conf = self._predict(frame)
                    your_label = self._canonical(idx)
                    comp_label = self._computer_choice()
                    winner = self._get_winner(comp_label, your_label)

                    print("-" * 78)
                    print(
                        f"Your choice: {self.emoji.get(your_label, '✌️')} ({your_label})  |  "
                        f"Computer: {self.emoji.get(comp_label, '✌️')} ({comp_label})  |  "
                        f"Conf: {conf:.2f}"
                    )
                    if winner == "tie":
                        print("😜 It's a tie!")
                    elif winner == "user":
                        print("😀 You win!")
                    else:
                        print("🙄 Computer wins!")

                elif key in (ord('q'), ord('Q'), 27):  # 27 = ESC
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
