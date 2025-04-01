# Hand Gesture Control

This project lets you control some things on your computer using your hand gestures and your webcam! It's like having a magic wand for your screen. 👋✨

## What you need

* A computer (Windows, Mac, or Linux)
* A webcam (built-in or external)
* Python 3 (it's like the language the computer understands)
* Some special tools (we'll help you install these!)

## How to get started

1.  **Make sure you have Python 3:** If you don't have it, you can ask a grown-up to help you download it from [python.org](https://www.python.org/).

2.  **Get the project files:** You probably already have them if you're reading this! Make sure all the `.py` files and the `requirements.txt` file are in the same folder.

3.  **Install the special tools:**
    * Open a terminal or command prompt (it's like a special window where you can type commands).
    * Go to the folder where you saved the project files.
    * Type this command and press Enter:
        ```bash
        pip install -r requirements.txt
        ```
        This will install all the things the project needs to work. It might take a little while, so be patient! 😊

## How to make the magic happen!

1.  **Plug in your webcam** if it's not built-in.

2.  **Open the terminal or command prompt again** and go back to your project folder.

3.  **Run the program!** Type this and press Enter:
    ```bash
    python main.py
    ```
    A window should pop up showing what your webcam sees.

4.  **Show your hand to the camera!** Try these cool gestures:
    * **Thumbs Up 👍:** Should make the volume go up.
    * **Thumbs Down 👎:** Should make the volume go down.
    * **Five Fingers Up (like you're saying "hi!") 👋:** Should play or pause a video you're watching (like on YouTube).
    * **Two Fingers Pointing Up (index and middle close together)✌️:** Should rewind the video a little bit.
    * **Three Fingers Pointing Up (index, middle, and ring close together)👌:** Should fast forward the video a little bit.
    * **Four Fingers Up (index, middle, ring, and pinky)🖐️:** Should make the video go to full screen.

5.  **To stop the program**, just press the `q` key on your keyboard while the webcam window is open.

## What are these files?
```
Here's how the files in the project are organized:

```
- hand-sign-control-ML/
- ├── README.md          # This file! It explains the project.
- ├── base.py            # A simple example for hand tracking.
- ├── main.py            # The main program for hand gesture control.
- └── requirements.txt   # A list of Python dependencies.
```
```


* `hand-sign-control-ML/`: This is the main folder that holds everything.
    * `README.md`: That's this file! It tells you all about the project.
    * `base.py`: This is a simple example that shows how to see your hand on the screen. It doesn't do any controlling.
    * `main.py`: This is the main program that does the hand tracking and makes things happen on your computer.
    * `requirements.txt`: This file is like a shopping list for Python, telling it what extra tools to install.

## How to run this project

1.  **Open your terminal or command prompt.**
2.  **Navigate to the `hand-sign-control-ML` folder.** You can use the `cd` command to change directories. For example, if the folder is on your Desktop, you might type:
    ```bash
    cd Desktop/hand-sign-control-ML
    ```
    (The exact command might be a little different depending on where you saved the folder.)
3.  **Once you are inside the `hand-sign-control-ML` folder, run the main program** using this command:
    ```bash
    python main.py
    ```

## Things to remember

* Make sure the camera can see your hand clearly. Good lighting helps! 💡
* Hold your hand gesture steady for a moment so the computer can see it.
* Sometimes it might not work perfectly – that's okay! It's like learning a new trick. 😉
* Have fun trying it out! 🎉

This project uses some neat stuff like **MediaPipe** (to see your hand) and **PyAutoGUI** (to control your computer). If you're curious, you can ask a grown-up to help you search for them online to learn more! This is a cool way to see how computers can understand what we do with our bodies. 😊
