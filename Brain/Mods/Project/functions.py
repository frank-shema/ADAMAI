import os
import pyautogui
import time
from PIL import Image

# Define paths to store screenshots
connect_window_screenshot_path = "Mods/Project/data/connect\_window_screenshot.png"
cropped_display_area_path = "Mods/Project/data/cropped_display_area.png"


def capture_display_area():
    """Captures the area around the mouse click in the Connect window."""
    # Open the Connect window
    pyautogui.hotkey('win', 'k')
    time.sleep(2)  # Wait for the window to open

    # Prompt the user to click on their display and capture their mouse position
    print("Please click on the display you want to set as default.")
    print("You have 10 seconds to click on the display.")
    time.sleep(10)  # Give the user 10 seconds to click

    # Capture the entire screen
    full_screenshot = pyautogui.screenshot()
    full_screenshot.save(connect_window_screenshot_path)

    # Get the mouse position
    x, y = pyautogui.position()
    print(f"Mouse position: ({x}, {y})")

    # Define the area around the click (adjust these values as needed)
    crop_width, crop_height = 300, 100  # Example dimensions, adjust as needed
    crop_area = (x - crop_width // 2, y - crop_height // 2, x + crop_width // 2, y + crop_height // 2)
    print(f"Cropping area: {crop_area}")

    # Crop the screenshot around the click
    cropped_image = full_screenshot.crop(crop_area)
    # Save the cropped image
    cropped_image.save(cropped_display_area_path)
    print(f"Cropped area saved to {cropped_display_area_path}")


def reconnect_to_display():
    """Re-connects to the display using the saved cropped image."""
    # Open the Connect window
    pyautogui.hotkey('win', 'k')
    time.sleep(2)  # Wait for the window to open

    # Locate the saved image of the display area
    if not os.path.exists(cropped_display_area_path):
        print(f"File {cropped_display_area_path} does not exist.")
        return

    print(f"Attempting to locate {cropped_display_area_path} on the screen...")
    try:
        location = pyautogui.locateOnScreen(cropped_display_area_path, confidence=0.4)

        if location:
            # Click the center of the located image
            pyautogui.click(pyautogui.center(location))
            print(f"Reconnected to display using saved image {cropped_display_area_path}")
        else:
            print(f"Could not locate the saved display area on the screen.")
    except Exception as e:
        print(f"Error locating image: {e}")


def manage_display_connection():
    """Checks if the cropped image exists and calls appropriate function."""
    if os.path.exists(cropped_display_area_path):
        print(f"{cropped_display_area_path} exists. Attempting to reconnect...")
        reconnect_to_display()
    else:
        print(f"{cropped_display_area_path} does not exist. Capturing new display area...")
        capture_display_area()


# Example Usage:
#manage_display_connection()