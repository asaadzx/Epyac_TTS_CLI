from transformers import AutoProcessor, BarkModel
import scipy.io.wavfile
import torch
import time
import sys
import os

class ConsoleProgressBar:
    """Simple console-based progress bar"""
    def __init__(self, total=100, prefix='Progress:', suffix='Complete', length=50, fill='█'):
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.length = length
        self.fill = fill
        self.start_time = time.time()
        
    def update(self, iteration, additional_info=""):
        percent = ("{0:.1f}").format(100 * (iteration / float(self.total)))
        filled_length = int(self.length * iteration // self.total)
        bar = self.fill * filled_length + '-' * (self.length - filled_length)
        elapsed_time = time.time() - self.start_time
        progress_line = f'\r{self.prefix} |{bar}| {percent}% {self.suffix} [{elapsed_time:.1f}s] {additional_info}'
        sys.stdout.write(progress_line)
        sys.stdout.flush()
        if iteration >= self.total:
            sys.stdout.write('\n')
            sys.stdout.flush()

def generate_speech(text, voice_preset, output_file):
    """Generate speech using Bark TTS with a progress bar"""
    print("\nStarting Bark Text-to-Speech Generation")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available. Using CPU (this will be slower)")

    progress = ConsoleProgressBar(total=100, prefix='Loading Model:', suffix='Complete')
    progress.update(10, "Loading Bark processor...")
    processor = AutoProcessor.from_pretrained("suno/bark")
    
    progress.update(30, "Loading Bark model (with CUDA)...")
    model = BarkModel.from_pretrained("suno/bark").to(device)
    progress.update(70, f"Model loaded on {device.upper()} successfully")
    
    progress.update(80, "Processing text input...")
    inputs = processor(text, voice_preset=voice_preset)
    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    
    progress = ConsoleProgressBar(total=100, prefix='Generating Speech:', suffix='Complete')
    progress.update(10, "Starting generation (this may take a while)...")
    audio_array = model.generate(**inputs)
    progress.update(80, "Audio generated successfully")
    
    audio_array = audio_array.cpu().numpy().squeeze()
    progress.update(90, f"Saving audio to {output_file}...")
    sample_rate = model.generation_config.sample_rate
    scipy.io.wavfile.write(output_file, rate=sample_rate, data=audio_array)
    
    progress.update(100, "Complete!")
    print(f"\nAudio saved to: {output_file}")
    
    return audio_array, sample_rate

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    # List of available voice presets (example set; adjust based on Bark's actual options)
    voice_presets = [
        "v2/en_speaker_1", "v2/en_speaker_2", "v2/en_speaker_3",
        "v2/en_speaker_4", "v2/en_speaker_5", "v2/en_speaker_6",
        "v2/en_speaker_7", "v2/en_speaker_8", "v2/en_speaker_9"
    ]

    # Improved CLI UI
    clear_screen()
    print("=====================================")
    print("    Bark Text-to-Speech Generator    ")
    print("=====================================")
    print("Welcome! Convert your text to speech with ease.\n")

    # Get text input Using CPU (this will be slower)"
    print("Step 1: Enter the text you want to convert to speech")
    text = input("Your text: ").strip()
    while not text:
        print("Text cannot be empty. Please try again.")
        text = input("Your text: ").strip()

    # Display voice options and get selection
    print("\nStep 2: Choose a voice preset")
    for i, preset in enumerate(voice_presets, 1):
        print(f"{i}. {preset}")
    while True:
        try:
            choice = int(input("Enter the number of your choice (1-9): "))
            if 1 <= choice <= len(voice_presets):
                voice_preset = voice_presets[choice - 1]
                break
            else:
                print(f"Please enter a number between 1 and {len(voice_presets)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Get output file path and name
    print("\nStep 3: Specify where to save the audio file")
    print("Example: /path/to/save/my_audio.wav or my_audio.wav")
    output_file = input("File path and name: ").strip()
    while not output_file.endswith('.wav'):
        print("File must end with '.wav'. Please try again.")
        output_file = input("File path and name: ").strip()
    
    # Ensure the directory exists
    output_dir = os.path.dirname(output_file) or '.'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Generate the speech
    print("\nGenerating your audio...")
    print("-------------------------------------")
    generate_speech(text, voice_preset, output_file)
    print("-------------------------------------")
    print("Thank you for using Sharkos TTS Generator with Bark Model!")

if __name__ == "__main__":
    main()