import os
import subprocess

def is_mp4_h264_aac(video_path):
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=codec_name', '-of', 'default=nokey=1:noprint_wrappers=1', video_path],
            capture_output=True, text=True, check=True)
        video_codec = result.stdout.strip()

        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries', 'stream=codec_name', '-of', 'default=nokey=1:noprint_wrappers=1', video_path],
            capture_output=True, text=True, check=True)
        audio_codec = result.stdout.strip()

        return video_codec == 'h264' and audio_codec == 'aac'
    except subprocess.CalledProcessError:
        return False

def reencode_videos(directory):
    # Supported video extensions
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm']

    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                input_path = os.path.join(root, file)
                
                if is_mp4_h264_aac(input_path):
                    print(f"Video already in correct format: {input_path}")
                    continue

                output_path = os.path.join(root, 'temp_' + file)

                # Construct the ffmpeg command
                command = [
                    'ffmpeg',
                    '-i', input_path,
                    '-vcodec', 'libx264',
                    '-acodec', 'aac',
                    '-strict', 'experimental',
                    '-y', output_path  # -y flag to overwrite without asking
                ]

                # Run the ffmpeg command
                try:
                    subprocess.run(command, check=True)
                    print(f"Successfully re-encoded: {input_path}")

                    # Replace the original file with the re-encoded file
                    os.replace(output_path, input_path)
                    print(f"Replaced original file with re-encoded file: {input_path}")
                except subprocess.CalledProcessError as e:
                    print(f"Failed to re-encode {input_path}: {e}")

if __name__ == "__main__":
    # Directory to start the recursive search
    directory = "."
    reencode_videos(directory)
