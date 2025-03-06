from transformers import AutoProcessor, BarkModel
import scipy.io.wavfile
import torch
import time
import sys
import os
from colorama import init, Fore, Style

# Initialize colorama for cross-platform color support
init()

class ConsoleProgressBar:
    """Simple console-based progress bar with colors"""
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
        bar = f"{Fore.GREEN}{self.fill * filled_length}{Style.RESET_ALL}" + '-' * (self.length - filled_length)
        elapsed_time = time.time() - self.start_time
        progress_line = f'\r{self.prefix} |{bar}| {Fore.RED}{percent}%{Style.RESET_ALL} {self.suffix} [{elapsed_time:.1f}s] {additional_info}'
        sys.stdout.write(progress_line)
        sys.stdout.flush()
        if iteration >= self.total:
            sys.stdout.write('\n')
            sys.stdout.flush()

def generate_speech(text, voice_preset, output_file):
    """Generate speech using Bark TTS with a progress bar"""
    print(f"\n{Fore.BLUE}Starting Epyac Text-to-Speech Generation{Style.RESET_ALL}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {Fore.YELLOW}{device.upper()}{Style.RESET_ALL}")
    
    progress = ConsoleProgressBar(total=100, prefix=f'{Fore.CYAN}Loading Model:{Style.RESET_ALL}')
    progress.update(10, "Loading Bark processor...")
    processor = AutoProcessor.from_pretrained("suno/bark")
    progress.update(30, "Loading Bark model...")
    model = BarkModel.from_pretrained("suno/bark").to(device)
    progress.update(70, f"Model loaded on {device.upper()}")
    
    progress.update(80, "Processing text input...")
    inputs = processor(text, voice_preset=voice_preset)
    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    
    progress = ConsoleProgressBar(total=100, prefix=f'{Fore.CYAN}Generating Speech:{Style.RESET_ALL}')
    progress.update(10, "Starting generation...")
    audio_array = model.generate(**inputs)
    progress.update(80, "Audio generated")
    
    audio_array = audio_array.cpu().numpy().squeeze()
    progress.update(90, f"Saving to {output_file}...")
    sample_rate = model.generation_config.sample_rate
    scipy.io.wavfile.write(output_file, rate=sample_rate, data=audio_array)
    
    # set environment variables for faster generation 
    os.environ["SUNO_OFFLOAD_CPU"] = "True"
    os.environ["SUNO_USE_SMALL_MODELS"] = "True"

    progress.update(100, "Complete!")
    print(f"\n{Fore.GREEN}Audio saved to: {output_file}{Style.RESET_ALL}")
    
    return audio_array, sample_rate

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    # Multi-language voice presets (hypothetical, expandable based on Bark’s capabilities)
    voice_presets = {
        "English": ["v2/en_speaker_1", "v2/en_speaker_2", "v2/en_speaker_3", "v2/en_speaker_4", "v2/en_speaker_5", "v2/en_speaker_6", "v2/en_speaker_7", "v2/en_speaker_8", "v2/en_speaker_9"],
        "German": ["v2/de_speaker_1", "v2/de_speaker_2", "v2/de_speaker_3", "v2/de_speaker_4"],
        "Spanish": ["v2/es_speaker_1", "v2/es_speaker_2", "v2/es_speaker_3", "v2/es_speaker_4", "v2/es_speaker_5"],
        "French": ["v2/fr_speaker_1", "v2/fr_speaker_2", "v2/fr_speaker_3", "v2/fr_speaker_4"],
        "Hindi": ["v2/hi_speaker_1", "v2/hi_speaker_2", "v2/hi_speaker_3", "v2/hi_speaker_4"],
        "Italian": ["v2/it_speaker_1", "v2/it_speaker_2", "v2/it_speaker_3"],
        "Japanese": ["v2/ja_speaker_1", "v2/ja_speaker_2", "v2/ja_speaker_3", "v2/ja_speaker_4", "v2/ja_speaker_5"],
        "Korean": ["v2/ko_speaker_1", "v2/ko_speaker_2", "v2/ko_speaker_3"],
        "Polish": ["v2/pl_speaker_1", "v2/pl_speaker_2", "v2/pl_speaker_3"],
        "Portuguese": ["v2/pt_speaker_1", "v2/pt_speaker_2", "v2/pt_speaker_3", "v2/pt_speaker_4"],
        "Russian": ["v2/ru_speaker_1", "v2/ru_speaker_2", "v2/ru_speaker_3", "v2/ru_speaker_4"],
        "Turkish": ["v2/tr_speaker_1", "v2/tr_speaker_2", "v2/tr_speaker_3"],
        "Chinese": ["v2/zh_speaker_1", "v2/zh_speaker_2", "v2/zh_speaker_3", "v2/zh_speaker_4", "v2/zh_speaker_5"]
    }

    # ASCII art for "Epyac_TTS"
    clear_screen()
    print(f"{Fore.CYAN}========================================{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}  {Style.RESET_ALL}")
    print(f"""{Fore.YELLOW} 
  

 /$$$$$$$$                                                     
| $$_____/                                                     
| $$        /$$$$$$  /$$   /$$  /$$$$$$   /$$$$$$$             
| $$$$$    /$$__  $$| $$  | $$ |____  $$ /$$_____/             
| $$__/   | $$  \ $$| $$  | $$  /$$$$$$$| $$                   
| $$      | $$  | $$| $$  | $$ /$$__  $$| $$                   
| $$$$$$$$| $$$$$$$/|  $$$$$$$|  $$$$$$$|  $$$$$$$             
|________/| $$____/  \____  $$ \_______/ \_______/             
          | $$       /$$  | $$                                 
          | $$      |  $$$$$$/                                 
          |__/       \______/                                  
 /$$$$$$$$ /$$$$$$$$ /$$$$$$         /$$$$$$  /$$       /$$$$$$
|__  $$__/|__  $$__//$$__  $$       /$$__  $$| $$      |_  $$_/
   | $$      | $$  | $$  \__/      | $$  \__/| $$        | $$  
   | $$      | $$  |  $$$$$$       | $$      | $$        | $$  
   | $$      | $$   \____  $$      | $$      | $$        | $$  
   | $$      | $$   /$$  \ $$      | $$    $$| $$        | $$  
   | $$      | $$  |  $$$$$$/      |  $$$$$$/| $$$$$$$$ /$$$$$$
   |__/      |__/   \______/        \______/ |________/|______/

                          
{Style.RESET_ALL}""")
    print(f"{Fore.CYAN}========================================{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Welcome to Epyac TTS Generator! 🎙️{Style.RESET_ALL}\n")

    # Step 1: Text input choice
    print(f"{Fore.GREEN}Step 1: How would you like to provide text?{Style.RESET_ALL}")
    print("1. Type it now")
    print("2. Load from a text file")
    while True:
        try:
            choice = int(input(f"{Fore.YELLOW}Your choice (1-2): {Style.RESET_ALL}"))
            if choice in [1, 2]:
                break
            print("Please enter 1 or 2.")
        except ValueError:
            print("Invalid input. Enter a number.")

    if choice == 1:
        text = input(f"{Fore.YELLOW}Enter your text: {Style.RESET_ALL}").strip()
        while not text:
            print("Text cannot be empty.")
            text = input(f"{Fore.YELLOW}Enter your text: {Style.RESET_ALL}").strip()
    else:
        file_path = input(f"{Fore.YELLOW}Enter text file path: {Style.RESET_ALL}").strip()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            if not text:
                raise ValueError("File is empty.")
        except Exception as e:
            print(f"{Fore.RED}Error: {e}. Please try again.{Style.RESET_ALL}")
            return

    # Step 2: Voice selection
    print(f"\n{Fore.GREEN}Step 2: Choose a voice preset{Style.RESET_ALL}")
    for lang, presets in voice_presets.items():
        print(f"{Fore.CYAN}{lang}:{Style.RESET_ALL}")
        for i, preset in enumerate(presets, 1):
            print(f"  {i}. {preset}")
    while True:
        try:
            lang_choice = input(f"{Fore.YELLOW}Enter language (e.g., English): {Style.RESET_ALL}").capitalize()
            if lang_choice in voice_presets:
                presets = voice_presets[lang_choice]
                choice = int(input(f"{Fore.YELLOW}Enter preset number (1-{len(presets)}): {Style.RESET_ALL}"))
                if 1 <= choice <= len(presets):
                    voice_preset = presets[choice - 1]
                    break
                print(f"Number must be between 1 and {len(presets)}.")
            else:
                print("Invalid language.")
        except ValueError:
            print("Invalid input. Try again.")

    # Step 3: Output file
    print(f"\n{Fore.GREEN}Step 3: Save your audio{Style.RESET_ALL}")
    output_file = input(f"{Fore.YELLOW}File path (e.g., output.wav): {Style.RESET_ALL}").strip()
    while not output_file.endswith('.wav'):
        print("File must end with '.wav'.")
        output_file = input(f"{Fore.YELLOW}File path: {Style.RESET_ALL}").strip()
    
    output_dir = os.path.dirname(output_file) or '.'
    os.makedirs(output_dir, exist_ok=True)

    # Generate the speech
    print(f"\n{Fore.CYAN}Generating audio...{Style.RESET_ALL}")
    generate_speech(text, voice_preset, output_file)
    print(f"{Fore.GREEN}All done! Thank you for using Epyac TTS!{Style.RESET_ALL}")

if __name__ == "__main__":
    main()