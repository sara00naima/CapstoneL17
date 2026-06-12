import math
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

from gui_backend import (
    AppState,
    SourceState,
    PANEL_BG,
    ACCENT,
    ACCENT2,
    TEXT,
    TEXT_DIM,
    CANVAS_BG,
    GRID_COL,
)


class SourceRow(tk.Frame):
    def __init__(self, parent, source: SourceState, scene_view, **kwargs):
        super().__init__(parent, bg=PANEL_BG, **kwargs)
        self.source = source
        self.scene_view = scene_view
        self._build()

    def _build(self):
        s = self.source
        col = s.color

        tk.Frame(self, bg=col, width=4).pack(side="left", fill="y", padx=(0, 6))

        tk.Label(
            self, text=s.name.upper(), bg=PANEL_BG, fg=col,
            font=("Helvetica", 9, "bold"), width=7, anchor="w"
        ).pack(side="left")

        self._mute_var = tk.BooleanVar(value=s.mute)
        self._mute_btn = tk.Checkbutton(
            self, text="M", variable=self._mute_var,
            bg=PANEL_BG, fg=TEXT_DIM, selectcolor="#3a1a1a",
            activebackground=PANEL_BG, activeforeground=ACCENT,
            font=("Helvetica", 8, "bold"), indicatoron=False,
            relief="flat", padx=4, command=self._on_mute
        )
        self._mute_btn.pack(side="left", padx=2)

        self._solo_var = tk.BooleanVar(value=s.solo)
        self._solo_btn = tk.Checkbutton(
            self, text="S", variable=self._solo_var,
            bg=PANEL_BG, fg=TEXT_DIM, selectcolor="#1a3a1a",
            activebackground=PANEL_BG, activeforeground="#81c784",
            font=("Helvetica", 8, "bold"), indicatoron=False,
            relief="flat", padx=4, command=self._on_solo
        )
        self._solo_btn.pack(side="left", padx=2)

        tk.Label(self, text="Gain", bg=PANEL_BG, fg=TEXT_DIM, font=("Helvetica", 7)).pack(side="left", padx=(6, 1))
        self._gain_var = tk.DoubleVar(value=s.gain_db)
        self._gain_sl = tk.Scale(
            self, variable=self._gain_var, from_=-24, to=6, resolution=0.5,
            orient="horizontal", length=80, bg=PANEL_BG, fg=TEXT,
            troughcolor=ACCENT2, highlightthickness=0, showvalue=False,
            command=self._on_gain
        )
        self._gain_sl.pack(side="left")
        self._gain_lbl = tk.Label(self, text="0.0 dB", bg=PANEL_BG, fg=TEXT, font=("Courier", 8), width=7)
        self._gain_lbl.pack(side="left")

        tk.Label(self, text="Az", bg=PANEL_BG, fg=TEXT_DIM, font=("Helvetica", 7)).pack(side="left", padx=(8, 1))
        self._az_lbl = tk.Label(self, text=f"{s.azimuth:+.0f}°", bg=PANEL_BG, fg=col, font=("Courier", 8), width=5)
        self._az_lbl.pack(side="left")

        tk.Label(self, text="El", bg=PANEL_BG, fg=TEXT_DIM, font=("Helvetica", 7)).pack(side="left", padx=(6, 1))
        self._el_var = tk.DoubleVar(value=s.elevation)
        self._el_sl = tk.Scale(
            self, variable=self._el_var, from_=-30, to=90, resolution=1,
            orient="horizontal", length=70, bg=PANEL_BG, fg=TEXT,
            troughcolor=ACCENT2, highlightthickness=0, showvalue=False,
            command=self._on_elevation
        )
        self._el_sl.pack(side="left")
        self._el_lbl = tk.Label(self, text="0°", bg=PANEL_BG, fg=TEXT, font=("Courier", 8), width=4)
        self._el_lbl.pack(side="left")

        tk.Button(
            self, text="…", bg=ACCENT2, fg=TEXT, relief="flat",
            font=("Helvetica", 8), padx=4, command=self._pick_wav
        ).pack(side="left", padx=(8, 0))

        self._file_lbl = tk.Label(self, text="no file", bg=PANEL_BG, fg=TEXT_DIM, font=("Helvetica", 7), width=14, anchor="w")
        self._file_lbl.pack(side="left", padx=2)

    def _on_mute(self):
        self.source.mute = self._mute_var.get()
        self._mute_btn.config(fg=ACCENT if self.source.mute else TEXT_DIM)
        self.scene_view.redraw()

    def _on_solo(self):
        self.source.solo = self._solo_var.get()
        self._solo_btn.config(fg="#81c784" if self.source.solo else TEXT_DIM)

    def _on_gain(self, _=None):
        v = self._gain_var.get()
        self.source.gain_db = v
        self._gain_lbl.config(text=f"{v:+.1f} dB")

    def _on_elevation(self, _=None):
        v = self._el_var.get()
        self.source.elevation = v
        self._el_lbl.config(text=f"{v:+.0f}°")
        self.scene_view.redraw()

    def _pick_wav(self):
        p = filedialog.askopenfilename(
            title=f"Select WAV for {self.source.name}",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if p:
            self.source.wav_path = p
            self._file_lbl.config(text=Path(p).name[-14:])

    def refresh_az(self):
        self._az_lbl.config(text=f"{self.source.azimuth:+.0f}°")


class SceneView(tk.Canvas):
    RADIUS = 180
    NODE_R = 10

    def __init__(self, parent, state: AppState, **kwargs):
        super().__init__(parent, bg=CANVAS_BG, highlightthickness=0, **kwargs)
        self.state = state
        self._rows = []
        self._drag = None

        self.bind("<Configure>", lambda _: self.redraw())
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)

    def set_rows(self, rows):
        self._rows = rows

    def _center(self):
        return self.winfo_width() / 2, self.winfo_height() / 2

    def _source_to_xy(self, source: SourceState):
        cx, cy = self._center()
        r = self.RADIUS * math.cos(math.radians(max(0, source.elevation)))
        angle_rad = math.radians(90 - source.azimuth)
        x = cx + r * math.cos(angle_rad)
        y = cy - r * math.sin(angle_rad)
        return x, y

    def _xy_to_az(self, x, y):
        cx, cy = self._center()
        dx, dy = x - cx, cy - y
        angle_rad = math.atan2(dy, dx)
        az = 90 - math.degrees(angle_rad)
        return (az + 180) % 360 - 180

    def redraw(self):
        self.delete("all")
        cx, cy = self._center()
        R = self.RADIUS

        for frac in (0.33, 0.66, 1.0):
            r = R * frac
            self.create_oval(cx - r, cy - r, cx + r, cy + r, outline=GRID_COL, width=1)

        for label, ax, ay, anchor in [
            ("FRONT", cx, cy - R - 14, "s"),
            ("BACK", cx, cy + R + 14, "n"),
            ("LEFT", cx - R - 14, cy, "e"),
            ("RIGHT", cx + R + 14, cy, "w"),
        ]:
            self.create_text(ax, ay, text=label, fill="#7f8c9b", font=("Helvetica", 7), anchor=anchor)

        self.create_line(cx - R - 5, cy, cx + R + 5, cy, fill=GRID_COL, width=1, dash=(4, 4))
        self.create_line(cx, cy - R - 5, cx, cy + R + 5, fill=GRID_COL, width=1, dash=(4, 4))

        self.create_oval(cx - 6, cy - 6, cx + 6, cy + 6, fill=ACCENT2, outline=ACCENT, width=2)
        self.create_text(cx, cy, text="👂", font=("Helvetica", 9))

        for i, src in enumerate(self.state.sources):
            if src.mute:
                continue
            x, y = self._source_to_xy(src)
            nr = self.NODE_R
            col = src.color

            self.create_line(cx, cy, x, y, fill=col, width=1, dash=(3, 3), tags=f"stem_{i}")
            self.create_oval(x - nr, y - nr, x + nr, y + nr, fill=col, outline="white", width=1.5, tags=(f"node_{i}", "node"))

            el_r = nr + 3 + (src.elevation / 90) * 10
            self.create_oval(x - el_r, y - el_r, x + el_r, y + el_r, outline=col, width=1, dash=(2, 3), tags=f"elring_{i}")

            self.create_text(x, y - nr - 7, text=src.name, fill=col, font=("Helvetica", 7, "bold"), tags=f"lbl_{i}")

    def _on_press(self, event):
        nr = self.NODE_R + 4
        for i, src in enumerate(self.state.sources):
            if src.mute:
                continue
            x, y = self._source_to_xy(src)
            if abs(event.x - x) < nr and abs(event.y - y) < nr:
                self._drag = i
                return
        self._drag = None

    def _on_drag(self, event):
        if self._drag is None:
            return
        src = self.state.sources[self._drag]
        src.azimuth = self._xy_to_az(event.x, event.y)
        self.redraw()
        if self._rows:
            self._rows[self._drag].refresh_az()

    def _on_release(self, _event):
        self._drag = None


class OutputPanel(tk.Frame):
    def __init__(self, parent, state: AppState, **kwargs):
        super().__init__(parent, bg=PANEL_BG, **kwargs)
        self.state = state
        self._build()

    def _build(self):
        s = self.state

        def section(text):
            tk.Label(self, text=text, bg=PANEL_BG, fg=ACCENT, font=("Helvetica", 8, "bold")).pack(anchor="w", pady=(10, 2), padx=10)

        def dim(text):
            tk.Label(self, text=text, bg=PANEL_BG, fg=TEXT_DIM, font=("Helvetica", 7)).pack(anchor="w", padx=10)

        section("SONG INPUT (for demixing)")
        frm = tk.Frame(self, bg=PANEL_BG)
        frm.pack(fill="x", padx=10)
        self._song_lbl = tk.Label(frm, text="no file selected", bg=PANEL_BG, fg=TEXT_DIM, font=("Helvetica", 8), anchor="w")
        self._song_lbl.pack(side="left", fill="x", expand=True)
        tk.Button(frm, text="Browse…", bg=ACCENT2, fg=TEXT, relief="flat", font=("Helvetica", 8), command=self._pick_song).pack(side="right")
        dim("Leave blank if loading individual stem WAVs in Sources panel.")

        section("RENDERER")
        self._renderer_var = tk.StringVar(value=s.renderer)
        renderers = [
            ("binaural", "Binaural  (HOA → HRTF)"),
            ("ls17_binaural", "LS17 → Binaural  (museum simulation)"),
            ("ls17", "LS17 decoded  (17-channel WAV)"),
        ]
        for val, label in renderers:
            tk.Radiobutton(
                self, text=label, variable=self._renderer_var, value=val,
                bg=PANEL_BG, fg=TEXT, selectcolor=ACCENT2,
                activebackground=PANEL_BG, activeforeground=TEXT,
                font=("Helvetica", 8), command=self._on_renderer
            ).pack(anchor="w", padx=14)

        section("SPEAKER LAYOUT")
        frm2 = tk.Frame(self, bg=PANEL_BG)
        frm2.pack(fill="x", padx=10)
        self._layout_lbl = tk.Label(frm2, text="default (museum 17ch)", bg=PANEL_BG, fg=TEXT_DIM, font=("Helvetica", 8), anchor="w")
        self._layout_lbl.pack(side="left", fill="x", expand=True)
        tk.Button(frm2, text="Load CSV…", bg=ACCENT2, fg=TEXT, relief="flat", font=("Helvetica", 8), command=self._pick_layout).pack(side="right")

        section("HRTF (SOFA file)")
        frm3 = tk.Frame(self, bg=PANEL_BG)
        frm3.pack(fill="x", padx=10)
        self._hrtf_lbl = tk.Label(frm3, text="default HRTF", bg=PANEL_BG, fg=TEXT_DIM, font=("Helvetica", 8), anchor="w")
        self._hrtf_lbl.pack(side="left", fill="x", expand=True)
        tk.Button(frm3, text="Load SOFA…", bg=ACCENT2, fg=TEXT, relief="flat", font=("Helvetica", 8), command=self._pick_hrtf).pack(side="right")

        section("HOA ORDER")
        frm4 = tk.Frame(self, bg=PANEL_BG)
        frm4.pack(fill="x", padx=10)
        tk.Label(frm4, text="Order:", bg=PANEL_BG, fg=TEXT, font=("Helvetica", 8)).pack(side="left")
        self._order_var = tk.IntVar(value=s.hoa_order)
        for o in (1, 2, 3):
            tk.Radiobutton(
                frm4, text=str(o), variable=self._order_var, value=o,
                bg=PANEL_BG, fg=TEXT, selectcolor=ACCENT2,
                activebackground=PANEL_BG, font=("Helvetica", 8),
                command=lambda: setattr(s, "hoa_order", self._order_var.get())
            ).pack(side="left", padx=6)

        section("OUTPUT DIRECTORY")
        frm5 = tk.Frame(self, bg=PANEL_BG)
        frm5.pack(fill="x", padx=10)
        self._outdir_lbl = tk.Label(frm5, text=str(s.out_dir), bg=PANEL_BG, fg=TEXT_DIM, font=("Helvetica", 7), anchor="w")
        self._outdir_lbl.pack(side="left", fill="x", expand=True)
        tk.Button(frm5, text="Change…", bg=ACCENT2, fg=TEXT, relief="flat", font=("Helvetica", 8), command=self._pick_outdir).pack(side="right")

    def _pick_song(self):
        p = filedialog.askopenfilename(
            title="Select song file",
            filetypes=[("Audio files", "*.wav *.mp3 *.flac"), ("All files", "*.*")]
        )
        if p:
            self.state.song_path = p
            self._song_lbl.config(text=Path(p).name)

    def _on_renderer(self):
        self.state.renderer = self._renderer_var.get()

    def _pick_layout(self):
        p = filedialog.askopenfilename(
            title="Select speaker layout",
            filetypes=[("CSV / JSON", "*.csv *.json"), ("All files", "*.*")]
        )
        if p:
            self.state.layout_path = p
            self._layout_lbl.config(text=Path(p).name)

    def _pick_hrtf(self):
        p = filedialog.askopenfilename(
            title="Select HRTF SOFA file",
            filetypes=[("SOFA files", "*.sofa"), ("All files", "*.*")]
        )
        if p:
            self.state.hrtf_path = p
            self._hrtf_lbl.config(text=Path(p).name)

    def _pick_outdir(self):
        p = filedialog.askdirectory(title="Select output directory")
        if p:
            self.state.out_dir = Path(p)
            self._outdir_lbl.config(text=p)


class StatusBar(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg="#0d1117", height=22, **kwargs)
        self._var = tk.StringVar(value="Ready.")
        tk.Label(self, textvariable=self._var, bg="#0d1117", fg=TEXT_DIM, font=("Helvetica", 8), anchor="w").pack(side="left", padx=10)

    def set(self, msg: str):
        self._var.set(msg)
        self.update_idletasks()