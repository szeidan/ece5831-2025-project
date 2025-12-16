# Teachable Machine - Image

## RockPaperScissors class

### `__init__(model_path='model/keras_model.h5', labels_path='model/labels.txt')`
Loads the Teachable Machine **Keras model** and parses **labels.txt** robustly (supports lines like `0 Rock` or `Rock`).  
Normalizes labels to lowercase and stores `['rock','paper','scissors', …]`.  
Sets the expected input size `(224, 224)` and defines a simple emoji map for display.

### `_canonical(label_or_idx)`
Utility that **normalizes any input** to one of `rock | paper | scissors`.  
Accepts an **index**, a **label string**, or even a **digit-string** (`"0"`).  
Also maps variants like `scissors1` / `scissors2` → **`scissors`** to keep game logic consistent.

### `_prep_frame(frame_bgr)`
Converts an OpenCV BGR frame to **RGB**, center-crops/fits to **224×224**, and normalizes pixels to **[-1, 1]** (the same preprocessing used by Teachable Machine). Returns a **(1, 224, 224, 3)** batch.

### `_predict(frame_bgr)`
Runs inference on a single webcam frame:
1) calls `_prep_frame`,  
2) feeds the batch to the model,  
3) returns the **top class index** and **confidence**.

### `predict_image(image_path)`
Convenience helper for notebooks/tests. Loads an image from disk, applies the same preprocessing, and returns **(canonical_label, confidence)**.

### `_computer_choice()`
Returns the computer’s move as a **label** (one of `rock | paper | scissors`).  
Choosing by **label** (not by index) avoids out-of-range errors when the label file has exactly three classes.

### `_get_winner(computer_choice, user_choice)`
Compares **canonical labels** and returns `'tie' | 'computer' | 'user'` using standard RPS rules  
(`rock→scissors`, `paper→rock`, `scissors→paper`).

### `play(camera_index=0)`
Starts the **webcam game**:
- Click the OpenCV window to focus it.  
- Press **`p`** (or **Space**) to capture a frame and classify your hand.  
- Press **`q`** (or **Esc**) to quit.  
Shows your choice, the computer’s choice, confidence, and the round result in the terminal.

---

## Demo Video

Here is my [demo video](https://youtu.be/ZCzy_gt-62E). Enjoy!
