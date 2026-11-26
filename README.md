# UnoNETv1: a tiny language model on an Arduino UNO

UnoNET is a small **language model that runs on an Arduino UNO**. You type the start
of an animal name, and the UNO finishes it.

```
> lio
  best:  lion
  maybe: lionfish | lion | lionfish
  (310 ms for the best guess, 62 ms per letter)
> tige
  best:  tiger
  maybe: tiger | tiger shark | tiger
```

Everything is written **from scratch**:

- The neural network is written in plain **NumPy**, including backpropagation and the Adam optimizer. It doesn't use PyTorch or TensorFlow.
- The Arduino code is plain C with **integer math only**. It doesn't use any ML library.

The whole model is a **GRU** (a type of recurrent neural network) with about **20,000 parameters**.
It is stored as **8-bit integers** in the UNO's flash memory, and it uses less than 1 KB of the UNO's 2 KB of RAM.

---

## Contents

1. [What you need](#1-what-you-need)
2. [Get the code](#2-get-the-code)
3. [Put the model on your Arduino (no Python needed)](#3-put-the-model-on-your-arduino-no-python-needed)
4. [Use it](#4-use-it)
5. [Train the model yourself (optional)](#5-train-the-model-yourself-optional)
6. [Use your own word list (optional)](#6-use-your-own-word-list-optional)
7. [Problems and fixes](#7-problems-and-fixes)
8. [How it works](#8-how-it-works)
9. [Files in this project](#9-files-in-this-project)

---

## 1. What you need

**Hardware**

- An **Arduino UNO**. An original board or a cheap clone both work.
- A **USB cable** that carries data. Some cheap cables only carry power. If your computer doesn't see the board, try another cable.

That's all. You don't need sensors, a breadboard or any wires.

**Software**

| What | Needed for | Where to get it |
|---|---|---|
| Arduino IDE 2 | putting the model on the UNO | https://www.arduino.cc/en/software |
| Python 3.9 or newer | only if you want to train the model yourself | https://www.python.org/downloads/ |
| NumPy | only if you want to train the model yourself | `pip install numpy` |
| Git | optional, for downloading the code | https://git-scm.com/downloads |

> The trained model is already included in this project. If you only want to **run** it on your
> Arduino, you only need the Arduino IDE. Python is only needed to **train** it again.

---

## 2. Get the code

### Step 2.1: Make a folder for the project

Make a new folder where you keep your projects, for example `Documents\projects`.

- **Windows:** open File Explorer, go to `Documents`, then right-click and choose **New → Folder**. Name it `projects`.
- **macOS / Linux:** open a terminal and run:
  ```bash
  mkdir -p ~/projects
  ```

### Step 2.2: Download UnoNET into that folder

Use **one** of these two ways.

**Way A: Download ZIP (easiest)**

1. On this project's GitHub page, click the green **Code** button.
2. Click **Download ZIP**.
3. Move the ZIP file into your `projects` folder.
4. Right-click it and choose **Extract All** (Windows) or double-click it (macOS).
5. You now have a folder like `projects/UnoNETv1`. (It may be called `UnoNETv1-main`; that's fine.)

**Way B: With Git**

Open a terminal (on Windows use **Command Prompt**, **PowerShell** or **Git Bash**) and run:

```bash
cd Documents/projects
git clone https://github.com/system-builder-priyam/UnoNETv1.git
cd UnoNETv1
```

---

## 3. Put the model on your Arduino (no Python needed)

### Step 3.1: Install the Arduino IDE

1. Download **Arduino IDE 2** from https://www.arduino.cc/en/software.
2. Install it with the default options. If Windows asks to install **drivers**, click **Install**.

### Step 3.2: Plug in the UNO

Connect the UNO to your computer with the USB cable. A green **ON** light on the board should turn on.

> **Clone boards:** many cheap UNO clones use a USB chip called **CH340**. If your computer
> doesn't see the board, install the CH340 driver (search "CH340 driver" + your operating system).

### Step 3.3: Open the sketch

1. Open the Arduino IDE.
2. Click **File → Open...**
3. Go to your project folder, then `firmware` → `unonet`, and open **`unonet.ino`**.

The IDE opens `unonet.ino` together with its other files as tabs: `model.h`, `unonet_core.h`,
`selftest.h` and `test_vectors.h`. That's correct. They all have to stay together in the
`unonet` folder.

> The folder name (`unonet`) must match the file name (`unonet.ino`). Don't rename one without the other.

### Step 3.4: Choose the board and the port

1. Click **Tools → Board → Arduino AVR Boards → Arduino Uno**.
2. Click **Tools → Port** and choose the port your UNO is on:
   - Windows: something like **COM3** or **COM7**
   - macOS: something like **/dev/cu.usbmodem...** or **/dev/cu.usbserial...**
   - Linux: something like **/dev/ttyACM0** or **/dev/ttyUSB0**

   Not sure which one it is? Unplug the UNO, look at the list, plug it back in, and pick the one that appeared.

### Step 3.5: Upload

1. Click the **Upload** button (the round button with an arrow **→**, top left).
2. Wait. The IDE first compiles the code, then sends it to the board. At the bottom you should see something like:
   ```
   Sketch uses 29444 bytes (91%) of program storage space. Maximum is 32256 bytes.
   Global variables use 506 bytes (24%) of dynamic memory, ...
   ```
   followed by **Done uploading**.

The model is now stored on the UNO. It stays there even after you unplug the board,
so you only need to upload again if you change the code or retrain the model.

---

## 4. Use it

### Step 4.1: Open the Serial Monitor

1. Click **Tools → Serial Monitor** (or the magnifying-glass icon at the top right).
2. At the bottom of the Serial Monitor, set:
   - the line ending to **Newline**
   - the speed to **115200 baud**
3. Press the **reset button** on the UNO (the small button near the USB port). You should see:
   ```
   === unoNET - animal name predictor ===
   int8 GRU, 64 hidden units, 20K params in flash
   Type the start of an animal name and press Enter.
   ...
   ```

### Step 4.2: Run the self-test

Type `/test` in the message box at the top and press **Enter**. The UNO compares its own answers with
the answers from the Python model. Every line should say **PASS**:

```
running self-test...
  PASS  hidden state + logits after '.ele'
  PASS  greedy  '' -> sand cat
  PASS  greedy  'ele' -> elephant
  ...
  PASS  sample  black panther
  ...
SELF-TEST PASSED - bit-exact with quantize.py
```

"Bit-exact" means the 8-bit chip computes **exactly the same numbers** as the Python version.

### Step 4.3: Try it

| Type this | What happens |
|---|---|
| `ele` | completes the name: best guess + 3 more ideas |
| `pen`, `hum`, `cro`, `sea ` | try any start of an animal name |
| *(empty line, just press Enter)* | 5 random animals |
| `/t 0.5` | sets the **temperature**. Lower = safer guesses, higher = more creative (and stranger) |
| `/t 0` | only show the best guess |
| `/test` | run the self-test |
| `/help` | show the commands |

The model only knows lowercase letters `a`–`z`, space and `-`. Capital letters are turned into
lowercase. Other characters are ignored.

---

## 5. Train the model yourself (optional)

You only need this part if you want to train the model again, or change it.

### Step 5.1: Install Python and NumPy

1. Install Python 3.9+ from https://www.python.org/downloads/.
   **Windows:** in the installer, tick **"Add Python to PATH"**.
2. Open a terminal **inside the project folder**.
   - Windows: open the `UnoNETv1` folder in File Explorer, click the address bar, type `cmd` and press Enter.
   - macOS / Linux: `cd ~/projects/UnoNETv1`
3. Install NumPy:
   ```bash
   pip install -r requirements.txt
   ```
   (On macOS / Linux you may need `pip3` and `python3` instead of `pip` and `python`.)

> Always run the scripts **from the project folder** (the one with `README.md` in it),
> because they look for `data/animals.txt` from there.

### Step 5.2: Run the scripts in order

Each script builds on the one before it. The first three are learning steps and are
not needed for the final model, but they show where the ideas come from.

| # | Command | What it does | What you should see |
|---|---|---|---|
| 1 | `python vocab.py` | the tokenizer: turns letters into numbers | `812` and the first 3 names |
| 2 | `python bigram.py` | the simplest model: counts which letter follows which | a strange name and a loss of about **2.56** |
| 3 | `python neural_bigram.py` | the same model, but **learned** with gradient descent | loss goes from **3.37** down to about **2.53** |
| 4 | `python gru.py` | **the real model**: checks its own math, trains, saves `model_float.npz` | `PASS`, loss falling to about **0.87**, example completions |
| 5 | `python quantize.py` | turns the model into 8-bit integers and writes the Arduino files | int8 loss about **0.87**, `wrote firmware/unonet/model.h ...` |

Step 4 takes about **1 minute** on a normal laptop. Training uses a fixed random seed, so you get
exactly the same model every time.

### Step 5.3: Upload the new model

`quantize.py` rewrote `firmware/unonet/model.h` and `firmware/unonet/test_vectors.h`.
Go back to [Step 3.3](#step-33-open-the-sketch) and upload again. Then run `/test` again.

### Step 5.4 (optional): Test the C code on your computer

If you have a C compiler (`gcc` or `clang`), you can run the exact same C code on your PC:

```bash
gcc -O2 -Wall -o pc_test/unonet_test pc_test/test_main.c
./pc_test/unonet_test ele kan pen
```

It runs the same self-test as `/test` on the UNO, then completes the words you give it.

---

## 6. Use your own word list (optional)

You can teach UnoNET something other than animals, like city names, Pokémon-style names or your friends' names.

1. Open `data/animals.txt` and replace the names with your own, **one per line**. Keep these rules:
   - only lowercase `a`–`z`, space and `-`
   - no empty lines and no duplicates
   - names no longer than about 20 letters
   - a few hundred to about a thousand lines works well
2. Run `python gru.py`, then `python quantize.py`.
3. Upload again ([Step 3.5](#step-35-upload)).

If you want to use other characters (like digits), add them to the `chars` list in `vocab.py`.
The model and the Arduino code adapt automatically. Each extra character costs about 260 bytes of flash.

---

## 7. Problems and fixes

| Problem | Fix |
|---|---|
| `Access is denied` or `port busy` when uploading | Another program is using the port. **Close the Serial Monitor** (and any other serial tool, e.g. in VS Code), then upload again. If it still fails, unplug the UNO for 3 seconds, or restart the Arduino IDE. |
| `not in sync` or `programmer is not responding` | Some clones have an older bootloader. Try **Tools → Processor → ATmega328P (Old Bootloader)** (choose **Arduino Nano** as the board to see this option). Also check that the right port is selected. |
| No port appears under **Tools → Port** | Try a different USB cable (it may be power-only). For clone boards, install the **CH340** driver. |
| The Serial Monitor shows strange symbols | Set the speed to **115200 baud**. |
| Typing a word does nothing | Set the line ending to **Newline** in the Serial Monitor. |
| Nothing appears after opening the Serial Monitor | Press the UNO's reset button. |
| `/test` shows **FAIL** | The model files don't match. Run `python quantize.py` again, then upload again. |
| `ModuleNotFoundError: No module named 'numpy'` | Run `pip install -r requirements.txt`. |
| `FileNotFoundError: data/animals.txt` | Run the scripts from the project folder, not from inside another folder. |
| `python` is not recognized (Windows) | Reinstall Python and tick **"Add Python to PATH"**, or try `py` instead of `python`. |

---

## 8. How it works

### The pipeline

```
data/animals.txt  ->  vocab.py  ->  gru.py (train)  ->  quantize.py (int8)  ->  model.h  ->  Arduino UNO
   812 names         letters to     float model,        8-bit model +          weights      "ele" -> "elephant"
                     numbers        about 20K params     C header files         in flash
```

### 1. Letters become numbers (tokenizer)

A neural network only understands numbers. Each character gets a fixed number (ID):

```
.  space  -   a   b   c   ...  z
0    1    2   3   4   5   ...  28
```

`.` is a special token that means "start of name" and "end of name".
For example, `"cat"` becomes `[0, 5, 3, 22]`.

### 2. The model predicts the next letter

Like every language model (including the big ones), UnoNET does one thing:
**given the letters so far, it gives a probability for each possible next letter**.
It picks one, adds it, and repeats until it predicts `.` (the end).

The model is a **GRU** (Gated Recurrent Unit). It keeps a **memory** of 64 numbers (the
*hidden state*), which it updates after each letter. That memory is how it knows that
`elepha` should continue with `nt`. Two "gates" decide how much of the old memory to keep and
how much new information to add.

| Model | Score (loss, lower is better) | What it means |
|---|---|---|
| Random guessing | 3.37 | picking between 29 letters blindly |
| Bigram (counting) | 2.56 | looks at only 1 previous letter |
| Neural bigram | 2.53 | the same, but learned |
| **GRU (float)** | **0.87** | remembers the whole name so far |
| **GRU (int8, on the UNO)** | **0.87** | almost no loss from going to 8 bits |

From the first 3 letters, the best guess gets the full name exactly right for 480 of the 812
animals. Many of the misses are cases where several animals start the same way (`sea ...`).

### 3. Making it fit on the UNO (quantization)

The UNO has **32 KB** of storage, **2 KB** of RAM, no floating-point hardware, and runs at 16 MHz. So:

- **8-bit weights.** Each weight is stored as a whole number from -127 to 127, plus one scale factor per block. 20,000 weights take about 20 KB.
- **Weights stay in flash.** They are read directly from flash (`PROGMEM`), so they use no RAM.
- **Integer-only math.** Multiplications use small integers. Scaling uses one multiply and one bit-shift.
- **Lookup tables.** `sigmoid`, `tanh` and `exp` are replaced by small tables with straight-line interpolation between entries.
- **Integer sampling.** Random sampling uses an integer random-number generator (xorshift32), so the PC and the UNO produce exactly the same "random" names from the same seed.

| Part | Size |
|---|---|
| Input weights (29 x 192) | 5,568 bytes |
| Memory weights (192 x 64) | 12,288 bytes |
| Output weights (29 x 64) | 1,856 bytes |
| Biases + lookup tables | about 1,600 bytes |
| **Whole sketch** | **29,444 bytes (91% of flash)** |
| RAM | 506 bytes for global variables, plus about 600 bytes while predicting |
| Speed | about **62 ms per letter** |

---

## 9. Files in this project

```
UnoNETv1/
├── README.md               this guide
├── requirements.txt        Python packages (just NumPy)
├── data/
│   └── animals.txt         812 animal names, one per line (the training data)
├── vocab.py                tokenizer: letters <-> numbers, batches, loading the data
├── bigram.py               learning step 1: counting model (the baseline)
├── neural_bigram.py        learning step 2: the same model learned with gradient descent
├── gru.py                  the real model: GRU forward + backward pass, gradient check, Adam training
├── model_float.npz         the trained model (made by gru.py)
├── quantize.py             int8 conversion, integer reference model, writes the Arduino files
├── firmware/
│   └── unonet/
│       ├── unonet.ino      the Arduino sketch (Serial Monitor interface)
│       ├── unonet_core.h   integer GRU inference in C (a copy of quantize.py's math)
│       ├── selftest.h      the /test command: checks the C code against Python's answers
│       ├── model.h         the int8 weights (made by quantize.py, don't edit by hand)
│       └── test_vectors.h  expected answers for the self-test (made by quantize.py)
└── pc_test/
    └── test_main.c         runs the same C code and self-test on a PC
```
