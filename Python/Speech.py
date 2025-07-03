import speech_recognition as sr
import svgwrite


cumulative_text = ""


def text_to_svg(text, filename="output.svg"):
    """Function to convert text to SVG format using Hershey fonts."""
    dwg = svgwrite.Drawing(filename, profile='tiny')


    dwg.add(dwg.text(text, insert=(10, 50), fill='black', font_size=20, font_family='HersheySimplex', stroke='black',
                     stroke_width=1))

    dwg.save()
    print(f"SVG file saved as {filename}")


def voice_to_text():
    """Continuously listens and appends recognized text until the stop command."""
    global cumulative_text
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Listening... Speak now. Say 'stop listening' to finalize.")

        while True:
            try:
                audio = recognizer.listen(source)
                text = recognizer.recognize_google(audio).lower()


                if "stop listening" in text:
                    print("Stop command received. Finalizing SVG file.")
                    text_to_svg(cumulative_text, filename="final_output.svg")
                    print("Final SVG file created. Goodbye!")
                    break


                if "start listening" in text or "stop listening" in text:
                    print(f"Ignored command: {text}")
                    continue


                cumulative_text += " " + text
                print(f"Recognized: {text}")
                print(f"Cumulative Text: {cumulative_text}")


                text_to_svg(cumulative_text)

            except sr.UnknownValueError:
                print("Sorry, I could not understand the audio. Please try again.")
            except sr.RequestError as e:
                print(f"Request error: {e}")


if __name__ == "__main__":
    cumulative_text = ""
    print("Say 'start listening' to begin or 'stop listening' to finalize.")
    print("Directly say 'stop listening' if you wish to exit without input.")
    voice_to_text()