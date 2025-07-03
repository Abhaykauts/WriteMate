# ✍️ WriteMate

> Empowering independence through assistive writing technology.

---

## 📌 Overview

**WriteMate** is a portable patent published assistive device that enables individuals with physical disabilities to **write independently** — without the need for human scribes. It captures input via **voice** or **Indian Sign Language (ISL)**, processes it using **machine learning on a Raspberry Pi 5**, and outputs the text using a **CNC machine** that physically writes it onto paper.

---

## 🚀 Features

- 🗣️ **Voice Input** — Converts spoken words into written text.
- ✋ **ISL Gesture Input** — Interprets Indian Sign Language via camera and ML.
- ✍️ **CNC-Based Writing** — Outputs recognized text onto paper using G-code.
- 🧠 **On-Device ML Processing** — Real-time processing using Raspberry Pi 5.
- 🌐 **Multilingual & Customizable** — Select language, pen type, and paper size.
- 💼 **Portable & Easy to Use** — Designed for use in exams, schools, homes, or public spaces.

---

## 🧠 Project Motivation

People with disabilities such as **blindness**, **mutism**, or **non-functional arms** often depend on others to write for them during exams or documentation tasks. This dependency:
- Reduces independence
- Risks miscommunication
- Limits dignity

**WriteMate** eliminates these barriers by offering a self-reliant writing solution.

---

## 🎯 Objectives

- ✅ Design a self-contained system for independent writing.
- ✅ Real-time input processing using ML on Raspberry Pi 5.
- ✅ Integrate a CNC machine for text output on paper.
- ✅ Allow flexible controls for pen type, paper size, and language.
- ✅ Promote dignity and independence for differently-abled individuals.

---

## ⚙️ System Architecture

```text
[ Voice Input / ISL Gestures ]
            ↓
 [ ML Processing on Raspberry Pi 5 ]
            ↓
 [ Text Conversion ]
            ↓
 [ G-code Generation ]
            ↓
 [ CNC Machine Writes on Paper ]

## 👋 Gesture A
![Gesture A](Sign_Gestures/A.jpg)
