[README.md](https://github.com/user-attachments/files/21743469/README.md)
# Toxicity Detection System

## Installation

1. Install the required dependencies:
   - `tkinter`
   - `pandas`
   - `numpy`
   - `pickle`
   - `nltk`
   - `scikit-learn`
   - `smtplib`
   - `ssl`
   - `email.message`
   - `pyautogui`
   - `pytesseract`
   - `PIL`
   - `yt_dlp`
   - `moviepy`
   - `speech_recognition`
   - `pydub`
   - `matplotlib`

2. Download and install Tesseract-OCR for your operating system:
   - Windows: https://github.com/UB-Mannheim/tesseract/wiki/Tesseract-Open-Source-OCR-Engine-(main-repository)
   - macOS: `brew install tesseract`
   - Linux: `sudo apt-get install tesseract-ocr`

3. Configure your email credentials in the `SENDER_EMAIL` and `SENDER_PASSWORD` variables.

## Usage

1. Run the `test 6.py` script to start the Toxicity Detection System.
2. Use the various analysis features provided in the GUI:
   - Analyze YouTube videos
   - Analyze text files
   - Analyze direct text input
   - Monitor the screen for toxic content
   - View analysis results and history
   - Clear all data

## API

The main class in the application is `AlignedToxicityGUI`, which provides the following methods:

- `predict_toxicity(text)`: Predicts whether the given text is toxic or not.
- `send_alert_email(subject, body)`: Sends an email alert with the provided subject and body.
- `toggle_screen_monitoring()`: Starts or stops the live screen monitoring process.
- `analyze_youtube_video_thread(youtube_url)`: Analyzes a YouTube video for toxic content.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please create a new issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Testing

The application includes built-in testing functionality for the screen monitoring and YouTube analysis features. You can test these features by using the corresponding buttons in the GUI.
