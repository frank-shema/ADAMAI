import Brain.Cerebrum
import threading
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.clock import mainthread
import chromadb  # Ensure chromadb is installed
from chromadb.config import Settings  # Import Settings from chromadb.config

# Initialize client
client = chromadb.PersistentClient(Settings(path="chroma_db"))

# Pass the client to Cerebrum
brain = Brain.Cerebrum(client)

# GUI class definition
class Main(BoxLayout):
    def __init__(self, **kwargs):
        super(Main, self).__init__(**kwargs)
        self.orientation = 'vertical'

        # Create buttons
        self.start_button = Button(text="Start Listening", on_press=self.start_listening)
        self.stop_button = Button(text="Stop Listening", on_press=self.stop_listening)

        # Initially add the Start Listening button
        self.add_widget(self.start_button)

        # Continuously check the global state to update buttons
        self.update_button_state()

    @mainthread
    def update_button_state(self):
        # Update the buttons based on the global state
        with brain.state_lock:
            if state == "listening":
                if self.start_button in self.children:
                    self.remove_widget(self.start_button)
                if self.stop_button not in self.children:
                    self.add_widget(self.stop_button)
            else:
                if self.stop_button in self.children:
                    self.remove_widget(self.stop_button)
                if self.start_button not in self.children:
                    self.add_widget(self.start_button)

        # Schedule another update after 500ms
        self.schedule_update()

    def schedule_update(self):
        from kivy.clock import Clock
        Clock.schedule_once(lambda dt: self.update_button_state(), 0.5)

    def start_listening(self, instance):
        global state
        with brain.state_lock:
            state = "listening"
        # Start the brain function in a separate thread
        threading.Thread(target=brain.listen, daemon=True).start()

    def stop_listening(self, instance):
        # Call the function to stop listening
        brain.stop()


# Main app class
class MyApp(App):
    def build(self):
        return Main()


if __name__ == '__main__':
    MyApp().run()