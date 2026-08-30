#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tkinter desktop app: load a traffic sign photo and classify it.

Adapted from the digit-recognizer GUI. The canvas-drawing input of the digit
version is gone -- you cannot usefully hand-draw a traffic sign -- and is
replaced by a file picker plus a preview of the preprocessed 32x32 crop the
network actually sees.

Example:
    python gui.py --model models/gtsrb_mlp
"""
import argparse
import os
import sys
import tkinter as tk
from tkinter.filedialog import askopenfilename

import numpy as np
from PIL import Image, ImageTk

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from dataset import IMG_SIZE, NUM_FEATURES, label_description, preprocess_image
from ffnn import predict
from model_utils import load_model

SUPPORTED = (".jpg", ".jpeg", ".png", ".ppm", ".bmp")


class TrafficSignApp:
    def __init__(self, root, model_path):
        model = load_model(model_path)
        if model is None:
            raise SystemExit("Could not read a model from '%s'. Train one first." % model_path)
        self.parameters = model["parameters"]
        self.labels = label_description()
        self.image_data = np.zeros((NUM_FEATURES, 1))

        root.title("Traffic Sign Recognizer")
        root.geometry("820x640")
        root.resizable(False, False)

        # ---- input frame ----
        image_frame = tk.Frame(root)
        image_frame.place(relx=0, rely=0, relwidth=1, relheight=0.62)

        tk.Label(image_frame, text="Input Image", font=("Helvetica", 22)).place(
            relx=0, rely=0, relwidth=1, relheight=0.1)

        self.original_pane = tk.Frame(image_frame, bd=1, relief="sunken")
        self.original_pane.place(relx=0.04, rely=0.12, relwidth=0.44, relheight=0.8)
        self.processed_pane = tk.Frame(image_frame, bd=1, relief="sunken")
        self.processed_pane.place(relx=0.52, rely=0.12, relwidth=0.44, relheight=0.8)

        self.original_label = tk.Label(self.original_pane, text="No image loaded",
                                       font=("Helvetica", 13))
        self.original_label.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.processed_label = tk.Label(self.processed_pane,
                                        text="Preprocessed %dx%d" % (IMG_SIZE, IMG_SIZE),
                                        font=("Helvetica", 13))
        self.processed_label.place(relx=0, rely=0, relwidth=1, relheight=1)

        # ---- action frame ----
        action_frame = tk.Frame(root)
        action_frame.place(relx=0, rely=0.62, relwidth=1, relheight=0.1)
        tk.Button(action_frame, text="Open Image...", font=("Helvetica", 13),
                  command=self.open_image).place(relx=0.1, rely=0.2,
                                                 relwidth=0.22, relheight=0.6)
        tk.Button(action_frame, text="Classify", font=("Helvetica", 13),
                  command=self.classify).place(relx=0.39, rely=0.2,
                                               relwidth=0.22, relheight=0.6)
        tk.Button(action_frame, text="Clear", font=("Helvetica", 13),
                  command=self.clear).place(relx=0.68, rely=0.2,
                                            relwidth=0.22, relheight=0.6)

        # ---- prediction frame ----
        pred_frame = tk.Frame(root)
        pred_frame.place(relx=0, rely=0.72, relwidth=1, relheight=0.28)
        self.first_label = tk.Label(pred_frame, text="", font=("Helvetica", 20))
        self.first_label.place(relx=0, rely=0.05, relwidth=1, relheight=0.3)
        self.first_prob = tk.Label(pred_frame, text="", font=("Helvetica", 14))
        self.first_prob.place(relx=0, rely=0.35, relwidth=1, relheight=0.2)
        self.second_label = tk.Label(pred_frame, text="", font=("Helvetica", 14))
        self.second_label.place(relx=0, rely=0.6, relwidth=1, relheight=0.2)
        self.second_prob = tk.Label(pred_frame, text="", font=("Helvetica", 12))
        self.second_prob.place(relx=0, rely=0.8, relwidth=1, relheight=0.2)

    # ---------------------------------------------------------------- actions
    def open_image(self):
        filepath = askopenfilename()
        if not filepath:
            return

        if not filepath.lower().endswith(SUPPORTED):
            self.original_label.config(
                image="", text="Unsupported file type.\nUse one of: %s" % ", ".join(SUPPORTED))
            return

        with Image.open(filepath) as handle:
            original = handle.copy()
            processed = preprocess_image(handle.copy())

        self.show(self.original_label, original.convert("RGB").resize((260, 260)))
        self.show(self.processed_label,
                  Image.fromarray(processed).resize((260, 260), Image.NEAREST))
        self.image_data = processed.reshape(-1, 1) / 255.

    def classify(self):
        if np.equal(self.image_data, np.zeros((NUM_FEATURES, 1))).all():
            self.first_label.config(text="Open an image first")
            return

        prediction = predict(self.image_data, self.parameters, second_guess=True)
        first_lbl, first_prob = prediction["First Prediction"]
        sec_lbl, sec_prob = prediction["Second Prediction"]

        self.first_label.config(text=self.labels[int(first_lbl)])
        self.first_prob.config(text="Confidence: %.2f%%  (class %d)"
                                    % (float(first_prob) * 100, int(first_lbl)))
        self.second_label.config(text="Second guess: %s" % self.labels[int(sec_lbl)])
        self.second_prob.config(text="Confidence: %.2f%%  (class %d)"
                                     % (float(sec_prob) * 100, int(sec_lbl)))

    def clear(self):
        self.original_label.config(image="", text="No image loaded")
        self.processed_label.config(image="",
                                    text="Preprocessed %dx%d" % (IMG_SIZE, IMG_SIZE))
        for widget in (self.first_label, self.first_prob,
                       self.second_label, self.second_prob):
            widget.config(text="")
        self.image_data = np.zeros((NUM_FEATURES, 1))

    @staticmethod
    def show(label, pil_image):
        rendered = ImageTk.PhotoImage(pil_image)
        label.config(image=rendered, text="")
        label.image = rendered  # keep a reference so Tk does not garbage collect it


def main():
    parser = argparse.ArgumentParser(description="Traffic sign recognizer GUI")
    parser.add_argument("--model", default="models/gtsrb_mlp")
    args = parser.parse_args()

    root = tk.Tk()
    TrafficSignApp(root, args.model)
    root.mainloop()


if __name__ == "__main__":
    main()
