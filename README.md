# hand-sign-control-ML

This project lets you control your computer using hand gestures! By using your webcam and some cool Artificial Intelligence (AI) magic, the program can understand certain hand movements and turn them into actions, like changing the volume or skipping video parts.

## What you need

* **A computer:** Windows, macOS, or Linux should work.
* **A webcam:** The one built into your laptop or an external one.
* **Python 3:** This is the programming language the project uses. You probably already have it, but if not, you can download it from [python.org](https://www.python.org/).
* **Some special tools (libraries):** These are like extra building blocks that help Python do specific tasks. You'll need to install them. Don't worry, it's easy!

## How to set it up

1.  **Download the project:** You've probably already done this if you're reading this file! Make sure you have all the files in the same folder.

2.  **Install the needed tools:** Open a terminal or command prompt on your computer. Then, go to the folder where you saved the project files. You can do this using the `cd` command (which stands for "change directory"). For example, if your folder is on your Desktop, you might type something like:
    ```bash
    cd Desktop/adithyenkandasamy-hand-sign-control-ml
    ```
    (The exact command might be a little different depending on your computer.)

3.  **Install the libraries:** Once you're in the project folder in your terminal, run this command:
    ```bash
    pip install -r requirements.txt
    ```
    This command tells Python to install all the libraries listed in the `requirements.txt` file. It might take a few minutes.

## How to run the project

1.  **Make sure your webcam is connected.**

2.  **Open a terminal or command prompt again** and go to the project folder (like you did in step 2 of the setup).

3.  **Run the main program:** Type this command and press Enter:
    ```bash
    python main.py
    ```
    This will start the hand tracking program.

4.  **A window will pop up** showing the video from your webcam.

5.  **Show your hand to the camera!** Try making the following gestures:
    * **Thumbs Up:** Should increase the volume.
    * **Thumbs Down:** Should decrease the volume.
    * **Five Fingers Up (Palm facing the camera):** Should play or pause a video (like on YouTube).
    * **Two Fingers (Index and Middle close together, pointing up):** Should rewind a video.
    * **Three Fingers (Index, Middle, and Ring close together, pointing up):** Should fast forward a video.
    * **Four Fingers Up (Index, Middle, Ring, and Pinky, thumb tucked in):** Should make a video full screen.
    * **Cross Your Hands:** Should close the current browser tab.
    * **Swing your hand quickly from side to side:** Should also close the current browser tab (this might be useful if the "Crossed Hands" gesture isn't working well).

6.  **To stop the program**, press the `q` key on your keyboard while the webcam window is open.

## What the code does

* **`README.md`:** This file you're reading right now! It gives you information about the project.
* **`base.py`:** This is a basic script that just shows you how to get the webcam working and track your hands using AI. It doesn't do any control actions, but it's a good way to see the hand tracking in action. You can run it with `python base.py`.
* **`main.py`:** This is the main program that does the hand tracking and controls your computer based on the gestures you make.
* **`requirements.txt`:** This file lists all the Python libraries that the project needs to run properly.

## Important things to keep in mind

* **Make sure your hand is clearly visible** to the webcam. Good lighting helps too!
* **The gestures need to be held steady for a moment** for the program to recognize them.
* **The program might not be perfect** and might sometimes misinterpret your gestures. This is normal for AI projects!
* **Be patient and have fun experimenting!**

## Want to learn more?

This project uses some cool technologies:

* **MediaPipe:** A set of tools from Google that helps with things like tracking hands, faces, and more in real-time.
* **OpenCV (cv2):** A library for working with images and videos.
* **PyAutoGUI:** A library that lets Python control your mouse and keyboard.

You can search for these online to find out more about how they work! This project is a great starting point for learning about computer vision and hand gesture recognition.