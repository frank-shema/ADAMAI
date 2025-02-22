# SpeechRecognition.py, By: Chance Brownfield
from Brain.utils import *
from tkinter import Tk, simpledialog, filedialog

from future.moves import tkinter

from Brain.utils import record_and_transcribe, async_record_and_transcribe


class SpeechRecognition():
    def __init__(self):
        pass

    async def APSR(self, wake_words=None, timer=None, word_count=None, gmm_dir=None, unknown_speakers=False, output="text"):
        async for result in async_record_and_transcribe(wake_words, timer, word_count, gmm_dir, unknown_speakers,
                                                        output):
            yield result

    def ASR(self, wake_words=None, timer=1, word_count=None, gmm_dir=None, unknown_speakers=False, output="text"):
        results = record_and_transcribe(wake_words, timer, word_count, gmm_dir, unknown_speakers, output)
        print(f"Results:{results}")
        return results

class InputRecognition:
    def __init__(self):
        pass
    def Speech(self, dialogue=None, need_audio=False):
        if dialogue:
            self.Text(dialogue)
        if need_audio:
            # ASR with custom ASR engine
            result_text, result_audio = record_and_transcribe(output="both")
            return result_text, result_audio
        else:
            # ASR without audio
            result_text = record_and_transcribe(output="text")
            return result_text

    def Text(self, dialogue=None, choice=None):
        root = Tk()
        root.withdraw()  # Hide the main window
        result = None
        if dialogue:
            if choice:
                result = simpledialog.askstring("Input", dialogue, initialvalue=choice[0], parent=root)
            else:
                result = simpledialog.askstring("Input", dialogue, parent=root)
        else:
            if choice:
                result = simpledialog.askstring("Input", "Click on your choice:", initialvalue=choice[0], parent=root)
            else:
                result = simpledialog.askstring("Input", "Enter text:", parent=root)
        return result

    def Path(self, dialogue=None, extension=None):
        root = Tk()
        root.withdraw()  # Hide the main window

        if dialogue:
            dialog_label = tkinter.Label(root, text=dialogue)
            dialog_label.pack()

        file_path = filedialog.askopenfilename()

        if extension and not file_path.endswith(extension):
            print(f"Selected file must have extension '{extension}'")
            file_path = None

        return file_path