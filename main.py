import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import pandas as pd
import numpy as np
import pickle
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import os
import threading
import platform
import time
from datetime import datetime

# Email imports
import smtplib
import ssl
from email.message import EmailMessage

# Screen monitoring and OCR imports
import pyautogui
import pytesseract
from PIL import Image

# YouTube transcription imports
import yt_dlp
import moviepy.editor as mp
import speech_recognition as sr
from pydub import AudioSegment

# Visualization imports
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class AlignedToxicityGUI:
    def __init__(self):
        self.data = []
        self.current_file = None
        self.analysis_history = []  # To store results for visualization

        # Screen Monitoring State
        self.monitoring = False
        self.monitoring_thread = None
        self.last_detected_text = ""
        self.session_start = datetime.now()

        # --- EMAIL CONFIGURATION ---
        # IMPORTANT: Fill in your details below.
        # For Gmail, you may need to generate an "App Password".
        # 1. Go to your Google Account settings.
        # 2. Go to "Security".
        # 3. Under "Signing in to Google", select "2-Step Verification" and follow the steps.
        # 4. Return to Security, select "App passwords".
        # 5. Generate a new password for this app and paste it here.
        self.SENDER_EMAIL = "saqlainsample5@gmail.com"  # Your email address
        self.SENDER_PASSWORD = "hlek ozin ykuj qrqc"  # Your generated app password
        self.RECEIVER_EMAIL = "naikmuhammadsaqlain@gmail.com"  # Email address to send alerts to

        # Create a directory for screenshots if it doesn't exist
        self.screenshot_dir = "bullying_detections"
        if not os.path.exists(self.screenshot_dir):
            os.makedirs(self.screenshot_dir)
            print(f"Created directory for evidence: ./{self.screenshot_dir}")

        # Load models and dependencies
        self.setup_ocr()
        self.load_existing_components()
        self.setup_aligned_gui()

    def setup_ocr(self):
        """
        Configures Tesseract OCR, attempting to find the executable on Windows.
        """
        print("INFO: Setting up OCR...")
        try:
            # For Windows, try to find the Tesseract executable in common install paths.
            if platform.system() == "Windows":
                tesseract_paths = [
                    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                    r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
                ]
                for path in tesseract_paths:
                    if os.path.exists(path):
                        pytesseract.pytesseract.tesseract_cmd = path
                        print(f"INFO: Tesseract executable found at: {path}")
                        break
                else:
                    print("WARNING: Tesseract executable not found in default paths.")

            # Test if OCR is working
            test_image = Image.new('RGB', (100, 30), color='white')
            pytesseract.image_to_string(test_image)
            self.ocr_available = True
            print("SUCCESS: OCR is configured and ready.")

        except Exception as e:
            self.ocr_available = False
            print(f"ERROR: OCR setup failed: {e}")
            messagebox.showwarning("OCR Warning",
                                   f"OCR functionality limited: {e}\n\nPlease install Tesseract-OCR for screen monitoring:\n"
                                   "https://github.com/UB-Mannheim/tesseract/wiki")

    def load_existing_components(self):
        """
        Loads the pre-trained machine learning model and vectorizer.
        Falls back to a keyword-based system if model files are not found.
        """
        print("INFO: Loading detection models...")
        try:
            # Attempt to load the trained model and vocabulary
            with open(r"stopwords.txt", 'r') as file:
                self.content_list = file.read().split('\n')

            self.trained_model = pickle.load(open('LinearSVC.pkl', 'rb'))
            self.vocab = pickle.load(open("tfidfVectorizer.pkl", "rb"))
            self.models_loaded = True
            print("✅ AI Models loaded successfully!")

        except Exception as e:
            messagebox.showwarning("Model Warning",
                                   f"Could not load AI models: {e}\n\nRunning in fallback keyword-detection mode.")
            self.models_loaded = False
            self.bullying_keywords = [
                'hate', 'stupid', 'idiot', 'loser', 'dumb', 'ugly', 'kill', 'die',
                'bully', 'worthless', 'pathetic', 'moron', 'shut up', 'go away',
                'nobody likes you', 'kill yourself', 'waste of space', 'retard',
                'freak', 'weirdo', 'failure', 'trash', 'garbage', 'useless'
            ]
            print("⚠️ Running in keyword detection mode (AI models not found)")

    def remove_pattern(self, input_txt, pattern):
        if isinstance(input_txt, str):
            r = re.findall(pattern, input_txt)
            for i in r:
                input_txt = re.sub(re.escape(i), '', input_txt)
            return input_txt
        else:
            return ''

    def preprocess_text(self, text):
        processed_text = self.remove_pattern(text, '@[\w]*')
        processed_text = re.sub('[^a-zA-Z#]', ' ', processed_text)
        processed_text = ' '.join([w for w in processed_text.split() if len(w) > 3])

        try:
            lemmatizer = nltk.stem.WordNetLemmatizer()
            words = processed_text.split()
            words = [lemmatizer.lemmatize(i) for i in words]
            processed_text = ' '.join(words)
        except:
            pass

        return processed_text

    def predict_toxicity(self, text):
        """
        Predicts toxicity and returns both a string result and a boolean.
        """
        if not text or len(text.strip()) < 3:
            return "Non-Bullying", False

        try:
            if self.models_loaded:
                processed_text = self.preprocess_text(text)
                tfidf_vector = TfidfVectorizer(
                    stop_words=self.content_list,
                    lowercase=True,
                    vocabulary=self.vocab
                )
                preprocessed_data = tfidf_vector.fit_transform([processed_text])
                prediction = self.trained_model.predict(preprocessed_data)

                if prediction[0] == 1:
                    return "Bullying", True
                else:
                    return "Non-Bullying", False
            else:
                # Keyword-based fallback
                text_lower = text.lower()
                if any(keyword in text_lower for keyword in self.bullying_keywords):
                    return "Bullying", True
                return "Non-Bullying", False

        except Exception as e:
            return f"Error: {e}", False

    # --- Email Alert Method ---
    def send_alert_email(self, subject, body):
        """
        Sends an email alert in a separate thread to avoid blocking the GUI.
        """
        if self.SENDER_EMAIL == "your_email@gmail.com" or self.SENDER_PASSWORD == "your_app_password":
            print("WARNING: Email credentials not configured. Skipping email alert.")
            self.root.after(0, self.update_status, "⚠️ Email not sent: Please configure credentials.")
            return

        # Run the email sending task in a daemon thread
        email_thread = threading.Thread(target=self._send_email_task, args=(subject, body), daemon=True)
        email_thread.start()

    def _send_email_task(self, subject, body):
        """The actual email sending logic."""
        msg = EmailMessage()
        msg.set_content(body)
        msg['Subject'] = subject
        msg['From'] = self.SENDER_EMAIL
        msg['To'] = self.RECEIVER_EMAIL

        try:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
                smtp.login(self.SENDER_EMAIL, self.SENDER_PASSWORD)
                smtp.send_message(msg)
            print(f"SUCCESS: Email alert sent to {self.RECEIVER_EMAIL}")
            # Schedule GUI update on the main thread
            self.root.after(0, self.update_status, f"📧 Email alert sent to {self.RECEIVER_EMAIL}")
        except Exception as e:
            print(f"ERROR: Failed to send email: {e}")
            self.root.after(0, self.update_status, f"❌ Email alert failed: {e}")

    # --- Screen Monitoring Methods ---
    def toggle_screen_monitoring(self):
        """Starts or stops the screen monitoring thread."""
        if self.monitoring:
            self.stop_screen_monitoring()
        else:
            self.start_screen_monitoring()

    def start_screen_monitoring(self):
        """Starts the screen monitoring process."""
        if not self.ocr_available:
            messagebox.showerror("OCR Error", "Screen Monitoring requires a working Tesseract-OCR installation.")
            return

        self.monitoring = True
        self.screen_monitor_button.config(text="⏹️ STOP SCREEN MONITORING\n\n🔴 Actively scanning your screen",
                                          bg="#e53e3e", activebackground="#fc8181")
        self.update_status("👁️ Live screen monitoring started...")

        self.monitoring_thread = threading.Thread(target=self.monitor_screen_loop, daemon=True)
        self.monitoring_thread.start()

    def stop_screen_monitoring(self):
        """Stops the screen monitoring process."""
        self.monitoring = False
        self.screen_monitor_button.config(text="👁️ LIVE SCREEN MONITORING\n\n🔍 Scan screen for toxic text",
                                          bg="#dd6b20", activebackground="#f6ad55")
        self.update_status("🟢 System Ready - Screen monitoring stopped.")

    def monitor_screen_loop(self):
        """The main loop for monitoring the screen in a background thread."""
        while self.monitoring:
            try:
                screenshot = pyautogui.screenshot()
                extracted_text = pytesseract.image_to_string(screenshot).strip()

                if extracted_text and extracted_text != self.last_detected_text:
                    self.last_detected_text = extracted_text
                    cleaned_text = ' '.join(extracted_text.split())

                    if len(cleaned_text) > 10:
                        result_str, is_bullying = self.predict_toxicity(cleaned_text)

                        if is_bullying:
                            # Use root.after to schedule GUI updates on the main thread
                            self.root.after(0, self.handle_detection, result_str, cleaned_text)

                time.sleep(3)  # Check every 3 seconds
            except Exception as e:
                print(f"ERROR in monitoring loop: {e}")
                time.sleep(5)

    def handle_detection(self, result, text):
        """Handles GUI updates when bullying is detected on screen."""
        self.show_result_popup(result, text, "Live Screen Detection")
        self.update_results_display(result, "Live Screen", len(text))
        self.save_detection_screenshot()
        self.update_status(f"🚨 Bullying Detected on Screen!")

        # Send email alert
        subject = "🚨 Toxicity Alert: Harmful Content Detected on Screen"
        body = f"""
Hello,

This is an automated alert from the Toxicity Detection System.
Harmful content was detected on a monitored device.

Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Source: Live Screen Monitoring

Detected Text:
--------------------------------------------------
{text}
--------------------------------------------------

A screenshot has been saved to the 'bullying_detections' folder on the device for review.
"""
        self.send_alert_email(subject, body)

    def save_detection_screenshot(self):
        """Captures the current screen and saves it to the evidence directory."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detection_{timestamp}.png"
            filepath = os.path.join(self.screenshot_dir, filename)
            pyautogui.screenshot(filepath)
            print(f"SUCCESS: Evidence screenshot saved to '{filepath}'")
        except Exception as e:
            print(f"ERROR: Failed to save screenshot: {e}")
            messagebox.showerror("Screenshot Error", f"Failed to save screenshot: {e}")

    # --- YouTube Video Processing Methods ---
    def download_video(self, youtube_url, output_dir):
        print("📥 Downloading video...")
        ydl_opts = {
            'outtmpl': os.path.join(output_dir, '%(title)s [%(id)s].%(ext)s'),
            'format': 'mp4'
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            video_title = info.get('title', 'Unknown')
        video_files = [f for f in os.listdir(output_dir) if f.endswith('.mp4')]
        if video_files:
            return os.path.join(output_dir, video_files[-1]), video_title
        return None, None

    def extract_audio(self, video_path, audio_path):
        print("🎵 Extracting audio from video...")
        video = mp.VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, verbose=False, logger=None)
        video.close()
        print("✅ Audio extraction completed!")

    def chunk_audio(self, audio_path, chunk_length_ms=50000):
        print("✂️ Splitting audio into chunks...")
        audio = AudioSegment.from_wav(audio_path)
        chunks = []
        for i, start_time in enumerate(range(0, len(audio), chunk_length_ms)):
            chunk = audio[start_time:start_time + chunk_length_ms]
            chunk_path = f"chunk_{i}.wav"
            chunk.export(chunk_path, format="wav")
            chunks.append(chunk_path)
        return chunks

    def transcribe_audio_chunks(self, chunk_files):
        recognizer = sr.Recognizer()
        full_transcript = []
        for i, chunk_path in enumerate(chunk_files):
            self.root.after(0, lambda i=i: self.update_status(f"🎙️ Transcribing chunk {i + 1}/{len(chunk_files)}..."))
            print(f"🎙️ Processing chunk {i + 1}/{len(chunk_files)}...")
            try:
                with sr.AudioFile(chunk_path) as source:
                    recognizer.adjust_for_ambient_noise(source)
                    audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data)
                full_transcript.append(text)
                print(f"✅ Chunk {i + 1} completed")
            except sr.UnknownValueError:
                print(f"❓ Chunk {i + 1}: Could not understand audio")
                full_transcript.append("[UNINTELLIGIBLE]")
            except sr.RequestError as e:
                print(f"❌ Chunk {i + 1}: API error - {e}")
                full_transcript.append("[ERROR]")
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
        return full_transcript

    def analyze_youtube_video_thread(self, youtube_url):
        """Run YouTube analysis in separate thread"""
        try:
            output_dir = r"youtube_downloads"
            os.makedirs(output_dir, exist_ok=True)
            audio_path = os.path.join(output_dir, 'audio.wav')
            text_output_path = os.path.join(output_dir, 'youtube_transcript.txt')

            video_id = youtube_url.split('/')[-1].split('?')[0].split('=')[-1]
            existing_video = next(
                (os.path.join(output_dir, f) for f in os.listdir(output_dir) if video_id in f and f.endswith('.mp4')),
                None)

            if existing_video:
                self.root.after(0, lambda: self.update_status(
                    f"📁 Using existing video: {os.path.basename(existing_video)}"))
                video_path, video_title = existing_video, os.path.basename(existing_video).split(' [')[0]
            else:
                self.root.after(0, lambda: self.update_status("📥 Downloading video..."))
                video_path, video_title = self.download_video(youtube_url, output_dir)
                if not video_path:
                    self.root.after(0, lambda: messagebox.showerror("Error", "Could not download video!"))
                    return

            self.root.after(0, lambda: self.update_status("🎵 Extracting audio..."))
            self.extract_audio(video_path, audio_path)
            self.root.after(0, lambda: self.update_status("✂️ Splitting audio into chunks..."))
            chunk_files = self.chunk_audio(audio_path)
            self.root.after(0, lambda: self.update_status(f"🎙️ Starting transcription of {len(chunk_files)} chunks..."))
            full_transcript = self.transcribe_audio_chunks(chunk_files)
            final_transcript = " ".join(full_transcript)

            with open(text_output_path, 'w', encoding='utf-8') as f:
                f.write(f"Video Title: {video_title}\nURL: {youtube_url}\n---\n{final_transcript}")

            self.root.after(0, lambda: self.update_status("🔍 Analyzing transcript for toxicity..."))
            result, is_bullying = self.predict_toxicity(final_transcript)

            def update_gui():
                self.youtube_button.config(state='normal',
                                           text='🎥 ANALYZE YOUTUBE VIDEO\n\n📹 Download • Transcribe • Analyze')
                self.show_result_popup(result, final_transcript, f"YouTube Video: {video_title}")
                self.update_results_display(result, f"YouTube: {video_title}", len(final_transcript))
                if is_bullying:
                    subject = f"🚨 Toxicity Alert: Harmful Content in YouTube Video '{video_title}'"
                    body = f"""
Hello,

This is an automated alert from the Toxicity Detection System.
Harmful content was detected in a YouTube video transcript.

Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Source: YouTube Video Analysis
Video Title: {video_title}
Video URL: {youtube_url}

Detected Transcript Snippet:
--------------------------------------------------
{final_transcript[:1000]}...
--------------------------------------------------
"""
                    self.send_alert_email(subject, body)

            self.root.after(0, update_gui)
            if os.path.exists(audio_path): os.remove(audio_path)

        except Exception as e:
            def show_error():
                messagebox.showerror("Error", f"YouTube analysis failed: {str(e)}")
                self.update_status("❌ YouTube analysis failed")
                self.youtube_button.config(state='normal',
                                           text='🎥 ANALYZE YOUTUBE VIDEO\n\n📹 Download • Transcribe • Analyze')

            self.root.after(0, show_error)

    # --- GUI Setup and Helper Methods ---
    def create_aligned_button(self, parent, text, command, bg_color, hover_color, text_color="white"):
        button = tk.Button(parent, text=text, command=command, font=("Segoe UI", 12, "bold"), bg=bg_color,
                           fg=text_color, relief="raised", bd=3, cursor="hand2", width=35, height=6,
                           activebackground=hover_color, activeforeground=text_color, wraplength=280, justify="center")
        button.bind("<Enter>", lambda e, b=button, c=hover_color: b.config(bg=c, relief="raised", bd=4))
        button.bind("<Leave>", lambda e, b=button, c=bg_color: b.config(bg=c, relief="raised", bd=3))
        button.bind("<Button-1>", lambda e, b=button: b.config(relief="sunken", bd=2))
        button.bind("<ButtonRelease-1>", lambda e, b=button: b.after(150, lambda: b.config(relief="raised", bd=3)))
        return button

    def create_small_button(self, parent, text, command, bg_color, hover_color, text_color="white", width=20, height=2):
        button = tk.Button(parent, text=text, command=command, font=("Segoe UI", 10, "bold"), bg=bg_color,
                           fg=text_color, relief="raised", bd=2, cursor="hand2", width=width, height=height,
                           activebackground=hover_color, activeforeground=text_color)
        button.bind("<Enter>", lambda e, b=button, c=hover_color: b.config(bg=c))
        button.bind("<Leave>", lambda e, b=button, c=bg_color: b.config(bg=c))
        return button

    def setup_aligned_gui(self):
        self.root = tk.Tk()
        self.root.title("🛡️ TOXICITY DETECTION - VISUAL DASHBOARD")
        self.root.geometry("1400x950")
        self.root.configure(bg='#2d3748')
        self.root.resizable(True, True)
        self.root.minsize(1200, 800)
        main_container = tk.Frame(self.root, bg='#2d3748')
        main_container.pack(fill='both', expand=True, padx=20, pady=20)
        self.create_aligned_header(main_container)
        self.create_perfectly_aligned_buttons(main_container)
        self.create_aligned_results_section(main_container)
        self.create_aligned_status_section(main_container)

    def create_aligned_header(self, parent):
        header_frame = tk.Frame(parent, bg='#1a365d', relief='raised', bd=3)
        header_frame.pack(fill='x', pady=(0, 20), ipady=10)
        tk.Label(header_frame, text="🛡️ ADVANCED TOXICITY DETECTION SYSTEM", font=("Segoe UI", 24, "bold"),
                 bg='#1a365d', fg='#ffffff').pack(pady=(10, 5))
        tk.Label(header_frame, text="Grid Layout • Visual Dashboard • Professional Interface",
                 font=("Segoe UI", 12, "italic"), bg='#1a365d', fg='#a0aec0').pack(pady=(0, 10))

    def create_perfectly_aligned_buttons(self, parent):
        buttons_header_frame = tk.Frame(parent, bg='#2d3748')
        buttons_header_frame.pack(fill='x', pady=(0, 15))
        tk.Label(buttons_header_frame, text="🚀 ANALYSIS FEATURES - SELECT YOUR OPTION", font=("Segoe UI", 18, "bold"),
                 bg='#2d3748', fg='#e2e8f0').pack()

        main_button_container = tk.Frame(parent, bg='#2d3748')
        main_button_container.pack(expand=False, fill='x', pady=(0, 20))
        main_button_container.grid_columnconfigure(0, weight=1)

        center_frame = tk.Frame(main_button_container, bg='#2d3748')
        center_frame.grid(row=0, column=0)

        for i in range(3): center_frame.grid_columnconfigure(i, pad=15)
        for i in range(3): center_frame.grid_rowconfigure(i, pad=15)

        self.youtube_button = self.create_aligned_button(center_frame,
                                                         "🎥 ANALYZE YOUTUBE VIDEO\n\n📹 Download • Transcribe • Analyze",
                                                         self.show_youtube_dialog, "#c53030", "#fc8181")
        self.file_button = self.create_aligned_button(center_frame,
                                                      "📁 ANALYZE TEXT FILE\n\n📄 Upload • Process • Analyze",
                                                      self.show_file_dialog, "#2f855a", "#68d391")
        self.text_button = self.create_aligned_button(center_frame,
                                                      "📝 ANALYZE DIRECT TEXT\n\n✏️ Type • Paste • Analyze",
                                                      self.show_text_dialog, "#6b46c1", "#b794f6")
        self.screen_monitor_button = self.create_aligned_button(center_frame,
                                                                "👁️ LIVE SCREEN MONITORING\n\n🔍 Scan screen for toxic text",
                                                                self.toggle_screen_monitoring, "#dd6b20", "#f6ad55")
        self.results_button = self.create_aligned_button(center_frame,
                                                         "📊 VIEW RESULTS\n\n📈 Dashboard • History • Export",
                                                         self.show_results_dialog, "#b7791f", "#f6e05e")
        self.clear_all_button = self.create_aligned_button(center_frame,
                                                           "🗑️ CLEAR ALL DATA\n\n🧹 Reset • Clean • Fresh Start",
                                                           self.clear_all_data, "#718096", "#a0aec0")
        self.info_button = self.create_aligned_button(center_frame,
                                                      "📊 MODEL INFO & STATUS\n\n📈 View model accuracy\n🚪 Exit controls",
                                                      self.show_info_dialog, "#2b6cb0", "#4299e1")

        self.youtube_button.grid(row=0, column=0)
        self.file_button.grid(row=0, column=1)
        self.text_button.grid(row=0, column=2)
        self.screen_monitor_button.grid(row=1, column=0)
        self.results_button.grid(row=1, column=1)
        self.clear_all_button.grid(row=1, column=2)
        self.info_button.grid(row=2, column=1)

    def create_aligned_results_section(self, parent):
        results_frame = tk.Frame(parent, bg='#1a202c', relief='raised', bd=3)
        results_frame.pack(fill='both', expand=True, pady=(0, 20))
        results_frame.grid_columnconfigure(1, weight=1)
        results_frame.grid_rowconfigure(1, weight=1)
        results_header = tk.Frame(results_frame, bg='#1a202c')
        results_header.grid(row=0, column=0, columnspan=2, sticky='ew', pady=10)
        tk.Label(results_header, text="📊 ANALYSIS RESULTS DASHBOARD", font=("Segoe UI", 16, "bold"), bg='#1a202c',
                 fg='#e2e8f0').pack()

        text_content_frame = tk.Frame(results_frame, bg='#1a202c')
        text_content_frame.grid(row=1, column=0, sticky='ns', padx=20, pady=(0, 20))
        self.results_display = tk.Text(text_content_frame, width=60, font=("Segoe UI", 11), bg='#2d3748', fg='#e2e8f0',
                                       relief='sunken', bd=2, wrap=tk.WORD, state='disabled')
        self.results_display.pack(fill='both', expand=True)

        viz_frame = tk.Frame(results_frame, bg='#2d3748', relief='sunken', bd=2)
        viz_frame.grid(row=1, column=1, sticky='nsew', padx=(0, 20), pady=(0, 20))

        self.fig, self.ax = plt.subplots(facecolor='#2d3748')
        self.ax.set_facecolor('#4a5568')
        self.chart_canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.chart_canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)

        self.update_results_display(None, None, None, initial=True)
        self.update_results_visualization()

    def update_results_visualization(self):
        self.ax.clear()
        counts = {"Bullying": self.analysis_history.count("Bullying"),
                  "Non-Bullying": self.analysis_history.count("Non-Bullying")}
        bars = self.ax.bar(list(counts.keys()), list(counts.values()), color=['#e53e3e', '#38a169'])
        self.ax.set_title('Analysis History', color='white', fontsize=14, weight='bold')
        self.ax.set_ylabel('Count', color='white', fontsize=12)
        self.ax.tick_params(axis='x', colors='white', labelsize=11)
        self.ax.tick_params(axis='y', colors='white', labelsize=11)
        for spine in ['top', 'right']: self.ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']: self.ax.spines[spine].set_color('gray')
        self.ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='#718096')
        for bar in bars:
            yval = bar.get_height()
            self.ax.text(bar.get_x() + bar.get_width() / 2.0, yval, int(yval), va='bottom', ha='center', color='white',
                         weight='bold')
        self.fig.tight_layout()
        self.chart_canvas.draw()

    def create_aligned_status_section(self, parent):
        status_frame = tk.Frame(parent, bg='#1a202c', relief='raised', bd=2)
        status_frame.pack(fill='x', ipady=5)
        self.status_var = tk.StringVar(value="🟢 System Ready - Visual Dashboard Activated - Select Feature Above")
        tk.Label(status_frame, textvariable=self.status_var, font=("Segoe UI", 11, "bold"), bg='#1a202c', fg='#48bb78',
                 anchor='center').pack(expand=True)

    def show_youtube_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("🎥 YouTube Video Analysis")
        dialog.geometry("650x450")
        dialog.configure(bg='#2d3748')
        dialog.grab_set()
        dialog.update_idletasks()
        x, y = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2), (dialog.winfo_screenheight() // 2) - (
                    dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        header_frame = tk.Frame(dialog, bg='#c53030', height=100)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        tk.Label(header_frame, text="🎥 YOUTUBE VIDEO ANALYSIS", font=("Segoe UI", 18, "bold"), bg='#c53030',
                 fg='white').pack(expand=True)
        content_frame = tk.Frame(dialog, bg='#2d3748')
        content_frame.pack(fill='both', expand=True, padx=40, pady=30)
        tk.Label(content_frame, text="Enter YouTube URL:", font=("Segoe UI", 12, "bold"), bg='#2d3748',
                 fg='#e2e8f0').pack(anchor='center', pady=(0, 15))
        self.youtube_url_entry = tk.Entry(content_frame, font=("Segoe UI", 11), bg='#4a5568', fg='#e2e8f0',
                                          relief='solid', bd=2, insertbackground='#e2e8f0', width=60, justify='center')
        self.youtube_url_entry.pack(pady=(0, 30))
        tk.Label(content_frame,
                 text="📝 This will download the video, extract audio,\ntranscribe speech to text, and analyze for toxicity",
                 font=("Segoe UI", 10), bg='#2d3748', fg='#a0aec0', justify='center').pack(pady=(0, 30))
        button_frame = tk.Frame(content_frame, bg='#2d3748')
        button_frame.pack()
        self.create_small_button(button_frame, "🎥 ANALYZE VIDEO", lambda: self.start_youtube_analysis(dialog),
                                 "#c53030", "#fc8181", width=18, height=2).pack(side='left', padx=(0, 15))
        self.create_small_button(button_frame, "❌ CLOSE", dialog.destroy, "#718096", "#a0aec0", width=12,
                                 height=2).pack(side='left')

    def show_file_dialog(self):
        file_path = filedialog.askopenfilename(title="Select Text File for Analysis",
                                               filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    content = file.read()
                self.current_file = {'path': file_path, 'name': os.path.basename(file_path), 'content': content,
                                     'size': len(content)}
                self.update_status(f"📁 Analyzing file: {self.current_file['name']}")
                result, is_bullying = self.predict_toxicity(content)
                self.show_result_popup(result, content, f"File Analysis: {self.current_file['name']}")
                self.update_results_display(result, f"File: {self.current_file['name']}", len(content))
                self.update_status(f"✅ File analysis complete - {result}")
                if is_bullying:
                    subject = f"🚨 Toxicity Alert: Harmful Content in File '{self.current_file['name']}'"
                    body = f"""
Hello,

This is an automated alert from the Toxicity Detection System.
Harmful content was detected in an uploaded file.

Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Source: File Analysis
Filename: {self.current_file['name']}

Detected Text:
--------------------------------------------------
{content[:1000]}...
--------------------------------------------------
"""
                    self.send_alert_email(subject, body)
            except Exception as e:
                messagebox.showerror("File Error", f"Could not read file: {str(e)}")
                self.update_status("❌ File analysis failed")

    def show_text_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("📝 Text Analysis")
        dialog.geometry("750x550")
        dialog.configure(bg='#2d3748')
        dialog.grab_set()
        dialog.update_idletasks()
        x, y = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2), (dialog.winfo_screenheight() // 2) - (
                    dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        header_frame = tk.Frame(dialog, bg='#6b46c1', height=100)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        tk.Label(header_frame, text="📝 DIRECT TEXT ANALYSIS", font=("Segoe UI", 18, "bold"), bg='#6b46c1',
                 fg='white').pack(expand=True)
        content_frame = tk.Frame(dialog, bg='#2d3748')
        content_frame.pack(fill='both', expand=True, padx=40, pady=30)
        tk.Label(content_frame, text="Enter text to analyze:", font=("Segoe UI", 12, "bold"), bg='#2d3748',
                 fg='#e2e8f0').pack(anchor='center', pady=(0, 15))
        text_input = scrolledtext.ScrolledText(content_frame, height=12, font=("Segoe UI", 10), bg='#4a5568',
                                               fg='#e2e8f0', relief='solid', bd=2, wrap=tk.WORD,
                                               insertbackground='#e2e8f0')
        text_input.pack(fill='both', expand=True, pady=(0, 20))
        button_frame = tk.Frame(content_frame, bg='#2d3748')
        button_frame.pack()

        def analyze_text():
            text = text_input.get('1.0', tk.END).strip()
            if text:
                self.update_status("🔄 Analyzing text...")
                result, is_bullying = self.predict_toxicity(text)
                self.show_result_popup(result, text, "Direct Text Input")
                self.update_results_display(result, "Direct Text", len(text))
                self.update_status(f"✅ Text analysis complete - {result}")
                if is_bullying:
                    subject = "🚨 Toxicity Alert: Harmful Content Detected from Text Input"
                    body = f"""
Hello,

This is an automated alert from the Toxicity Detection System.
Harmful content was detected from a direct text input.

Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Source: Direct Text Input

Detected Text:
--------------------------------------------------
{text}
--------------------------------------------------
"""
                    self.send_alert_email(subject, body)
                dialog.destroy()
            else:
                messagebox.showwarning("Warning", "Please enter some text!")

        self.create_small_button(button_frame, "🔍 ANALYZE TEXT", analyze_text, "#6b46c1", "#b794f6", width=18,
                                 height=2).pack(side='left', padx=(0, 15))
        self.create_small_button(button_frame, "📄 LOAD SAMPLE", lambda: self.load_sample_text_dialog(text_input),
                                 "#2f855a", "#68d391", width=15, height=2).pack(side='left', padx=(0, 15))
        self.create_small_button(button_frame, "❌ CLOSE", dialog.destroy, "#718096", "#a0aec0", width=12,
                                 height=2).pack(side='left')

    def show_results_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("📊 Results Dashboard")
        dialog.geometry("900x650")
        dialog.configure(bg='#2d3748')
        dialog.grab_set()
        dialog.update_idletasks()
        x, y = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2), (dialog.winfo_screenheight() // 2) - (
                    dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        header_frame = tk.Frame(dialog, bg='#b7791f', height=100)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        tk.Label(header_frame, text="📊 ANALYSIS RESULTS DASHBOARD", font=("Segoe UI", 18, "bold"), bg='#b7791f',
                 fg='white').pack(expand=True)
        content_frame = tk.Frame(dialog, bg='#2d3748')
        content_frame.pack(fill='both', expand=True, padx=40, pady=30)
        results_display = tk.Text(content_frame, font=("Segoe UI", 11), bg='#4a5568', fg='#e2e8f0', relief='solid',
                                  bd=2, wrap=tk.WORD)
        results_display.pack(fill='both', expand=True, pady=(0, 20))
        results_display.insert('1.0', self.results_display.get('1.0', tk.END))
        results_display.config(state='disabled')
        self.create_small_button(content_frame, "❌ CLOSE", dialog.destroy, "#718096", "#a0aec0", width=15,
                                 height=2).pack()

    def show_info_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("📊 Model Information & Status")
        dialog.geometry("600x500")
        dialog.configure(bg='#2d3748')
        dialog.grab_set()
        dialog.resizable(False, False)
        dialog.update_idletasks()
        x, y = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2), (dialog.winfo_screenheight() // 2) - (
                    dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        header_frame = tk.Frame(dialog, bg='#2b6cb0', height=80)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        tk.Label(header_frame, text="📊 MODEL INFO & APP STATUS", font=("Segoe UI", 18, "bold"), bg='#2b6cb0',
                 fg='white').pack(expand=True)
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TNotebook", background='#2d3748', borderwidth=0)
        style.configure("TNotebook.Tab", background="#4a5568", foreground="#e2e8f0", font=("Segoe UI", 11, "bold"),
                        padding=[20, 8])
        style.map("TNotebook.Tab", background=[("selected", "#2b6cb0")], foreground=[("selected", "white")])
        notebook = ttk.Notebook(dialog, style="TNotebook")
        notebook.pack(expand=True, fill='both', padx=20, pady=20)

        accuracy_frame = tk.Frame(notebook, bg='#2d3748')
        notebook.add(accuracy_frame, text='📈 Model Accuracy')
        accuracy_content_frame = tk.Frame(accuracy_frame, bg='#2d3748')
        accuracy_content_frame.pack(expand=True, fill='both', padx=30, pady=20)
        tk.Label(accuracy_content_frame, text="Model Performance Metrics", font=("Segoe UI", 14, "bold"), bg='#2d3748',
                 fg='#4299e1').pack(pady=(10, 15))
        accuracy_text_content = "The toxicity detection model is a Linear Support Vector Classifier (LinearSVC).\n\nKey performance metrics on validation set:\n\n  •  Accuracy: ~95.6%\n  •  Precision (Bullying Class): ~94%\n  •  Recall (Bullying Class): ~91%"
        tk.Label(accuracy_content_frame, text=accuracy_text_content, font=("Segoe UI", 11), bg='#2d3748', fg='#a0aec0',
                 justify='left').pack(pady=(10, 15), anchor='w')

        status_frame = tk.Frame(notebook, bg='#2d3748')
        notebook.add(status_frame, text='⚙️ App Status & Exit')
        status_content_frame = tk.Frame(status_frame, bg='#2d3748')
        status_content_frame.pack(expand=True, fill='both', padx=30, pady=20)
        tk.Label(status_content_frame, text="Current Application Status", font=("Segoe UI", 14, "bold"), bg='#2d3748',
                 fg='#4299e1').pack(pady=(10, 15))
        tk.Label(status_content_frame, textvariable=self.status_var, font=("Segoe UI", 11, "italic"), bg='#4a5568',
                 fg='#e2e8f0', wraplength=450, relief='sunken', bd=2, padx=10, pady=10).pack(pady=(10, 40), fill='x')
        button_frame = tk.Frame(status_content_frame, bg='#2d3748')
        button_frame.pack()
        self.create_small_button(button_frame, "❌ EXIT APPLICATION", self.exit_application, "#e53e3e", "#fc8181",
                                 width=20, height=2).pack(side='left', padx=(0, 15))
        self.create_small_button(button_frame, "↩️ BACK", dialog.destroy, "#718096", "#a0aec0", width=15,
                                 height=2).pack(side='left')

    def start_youtube_analysis(self, dialog):
        youtube_url = self.youtube_url_entry.get().strip()
        if not youtube_url or not ("youtube.com" in youtube_url or "youtu.be" in youtube_url):
            messagebox.showwarning("Invalid URL", "Please enter a valid YouTube URL!")
            return
        dialog.destroy()
        self.youtube_button.config(state='disabled', text='🔄 PROCESSING YOUTUBE VIDEO...')
        self.update_status("🚀 Starting YouTube video analysis...")
        thread = threading.Thread(target=self.analyze_youtube_video_thread, args=(youtube_url,), daemon=True)
        thread.start()

    def load_sample_text_dialog(self, text_widget):
        sample_texts = ["You are such an idiot! I hate you and nobody likes you!",
                        "Have a wonderful day! You're amazing and I appreciate you!",
                        "Stop bothering me, you stupid loser! Get out of here!",
                        "Thanks for your help, I really appreciate your kindness.",
                        "You're so dumb, nobody wants to be around you!"]
        import random
        text_widget.delete('1.0', tk.END)
        text_widget.insert('1.0', random.choice(sample_texts))

    def clear_all_data(self):
        if messagebox.askyesno("Clear All", "Are you sure you want to clear all data, results, and visualizations?"):
            self.analysis_history.clear()
            self.update_results_display(None, None, None, initial=True)
            self.update_results_visualization()
            self.update_status("🧹 All data cleared - System ready for new analysis")

    def update_status(self, message):
        self.status_var.set(message)

    def update_results_display(self, result, source, length, initial=False):
        self.results_display.config(state='normal')
        self.results_display.delete('1.0', tk.END)
        if initial:
            self.results_display.insert('1.0',
                                        "📈 ANALYSIS RESULTS & HISTORY\n\n• Use any feature button to start analysis.\n• This panel shows the details of the latest analysis.\n• The chart to the right tracks the history of all results.\n\n🎯 Select a feature button to begin!")
            self.results_display.config(state='disabled')
            return

        self.analysis_history.append(result)
        self.update_results_visualization()

        status_icon, status_text = ("🚨", "TOXIC CONTENT DETECTED") if result == "Bullying" else ("🛡️",
                                                                                                 "CONTENT IS SAFE")
        result_text = f"📊 LATEST ANALYSIS RESULT\n\n{status_icon} STATUS: {status_text}\n📅 Analysis completed successfully\n📊 Source: {source}\n📏 Content Length: {length:,} characters\n🔍 Detection Result: {result}\n\n{'-' * 45}\n\n🎯 Analysis complete! The chart has been updated."
        self.results_display.insert('1.0', result_text)
        self.results_display.config(state='disabled')

    def show_result_popup(self, result, text_analyzed, analysis_type="Text"):
        popup = tk.Toplevel(self.root)
        popup.title("🎯 Analysis Result")
        popup.geometry("750x650")
        popup.configure(bg='#2d3748')
        popup.grab_set()
        popup.update_idletasks()
        x, y = (popup.winfo_screenwidth() // 2) - (popup.winfo_width() // 2), (popup.winfo_screenheight() // 2) - (
                    popup.winfo_height() // 2)
        popup.geometry(f"+{x}+{y}")

        header_bg, accent_color, result_text, icon = ('#c53030', '#fc8181', "⚠️ TOXIC CONTENT DETECTED",
                                                      "🚨") if result == "Bullying" else ('#2f855a', '#68d391',
                                                                                         "✅ CONTENT IS SAFE", "🛡️")

        header_frame = tk.Frame(popup, bg=header_bg, height=130)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        tk.Label(header_frame, text=icon, font=("Segoe UI", 48), bg=header_bg, fg='white').pack(expand=True,
                                                                                                pady=(20, 0))
        tk.Label(header_frame, text=result_text, font=("Segoe UI", 18, "bold"), bg=header_bg, fg='white').pack(
            expand=True, pady=(0, 20))

        content_frame = tk.Frame(popup, bg='#2d3748')
        content_frame.pack(fill='both', expand=True, padx=40, pady=30)
        tk.Label(content_frame, text=f"📊 Analysis Type: {analysis_type}", font=("Segoe UI", 14, "bold"), bg='#2d3748',
                 fg=accent_color).pack(anchor='center', pady=(0, 20))
        tk.Label(content_frame, text="📝 Analyzed Content:", font=("Segoe UI", 12, "bold"), bg='#2d3748',
                 fg='#e2e8f0').pack(anchor='center', pady=(0, 15))

        text_display = scrolledtext.ScrolledText(content_frame, height=12, font=("Consolas", 10), wrap=tk.WORD,
                                                 bg='#4a5568', fg='#e2e8f0', relief='solid', bd=2)
        text_display.pack(fill='both', expand=True, pady=(0, 25))
        display_text = text_analyzed[:2000] + "..." if len(text_analyzed) > 2000 else text_analyzed
        text_display.insert('1.0', display_text)
        text_display.config(state='disabled')

        self.create_small_button(content_frame, "✓ CLOSE", popup.destroy, accent_color,
                                 "#68d391" if result == "Non-Bullying" else "#fc8181", width=15, height=2).pack()

    def exit_application(self):
        if messagebox.askyesno("🚪 Exit Confirmation", "Are you sure you want to exit?"):
            if self.monitoring: self.stop_screen_monitoring()
            self.update_status("👋 Goodbye! Closing application...")
            self.root.after(500, self.root.destroy)

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.exit_application)
        self.root.update_idletasks()
        width, height = self.root.winfo_width(), self.root.winfo_height()
        x, y = (self.root.winfo_screenwidth() // 2) - (width // 2), (self.root.winfo_screenheight() // 2) - (
                    height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        self.root.mainloop()


if __name__ == "__main__":
    try:
        # Check for NLTK data, download if missing
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            print("INFO: First-time setup: Downloading NLTK 'wordnet' data for lemmatization...")
            nltk.download('wordnet')
        app = AlignedToxicityGUI()
        app.run()
    except KeyboardInterrupt:
        print("\n👋 Application terminated by user")
    except Exception as e:
        print(f"❌ Application error: {e}")
