from pynput import keyboard

# Define the variable to be updated when the target key is pressed
pressed_key = None

# Callback function for key press events
def on_press(key):
    global pressed_key
    pressed_key = key

# Create a keyboard listener
listener = keyboard.Listener(on_press=on_press)

# Start the listener
listener.start()

i = 1
while True:
    if pressed_key == keyboard.KeyCode.from_char('q'):
        break
    else:
        print(i)
        i+=1

listener.stop()