import yt_dlp
import os
from pydub import AudioSegment
import speech_recognition as sr

# Step 1: Download the YouTube Video
def download_youtube_video(url, output_dir="downloads"):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # yt-dlp options for audio download
    ydl_opts = {
        'format': 'bestaudio/best',
        'extractaudio': True,  # Only extract audio
        'audioquality': 1,     # Highest audio quality
        'outtmpl': f'{output_dir}/%(id)s.%(ext)s',  # Download path
    }

    # Download the audio
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        audio_file = f"{output_dir}/{info_dict['id']}.webm"  # The downloaded file is typically webm or audio format
        print(f"Downloaded audio file: {audio_file}")
        return audio_file

# Step 2: Split the audio into smaller chunks
def split_audio(audio_file, chunk_length_ms=60000):  # Default chunk length is 60 seconds (60000 ms)
    audio = AudioSegment.from_file(audio_file)  # Load the audio file
    chunks = []

    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]  # Extract the chunk
        chunk_name = f"chunk_{i // chunk_length_ms}.wav"
        chunk.export(chunk_name, format="wav")  # Save as WAV file
        chunks.append(chunk_name)

    return chunks

# Step 3: Transcribe Audio File to Text
def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()

    # Use the recognizer to convert speech to text
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        print("Recognizing...")

        # Recognize speech using Google Web API
        try:
            text = recognizer.recognize_google(audio_data)
            print("Transcription: " + text)
            return text
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand the audio.")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")

    return None

# Step 4: Main function to run the process
def fetch_transcript():
    youtube_url = input("Enter the YouTube video URL: ")

    # Step 1: Download the video
    audio_file = download_youtube_video(youtube_url)

    # Step 2: Split the audio into chunks
    chunks = split_audio(audio_file)

    # Step 3: Transcribe each chunk and combine the results
    transcriptions = []
    for chunk in chunks:
        transcription = transcribe_audio(chunk)
        if transcription:
            transcriptions.append(transcription)

        # Clean up the chunk after transcription
        os.remove(chunk)

    # Combine all transcriptions into one
    full_transcription = " ".join(transcriptions)
    if full_transcription:
        print("\nFull Transcription:\n")
        print(full_transcription)
        return full_transcription
    else:
        print("No transcription available.")
        return ""

    # Clean up the downloaded file
    os.remove(audio_file)