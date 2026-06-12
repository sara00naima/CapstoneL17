import math
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

from gui_backend import (
    AppState,
    SourceState,
    PANEL_BG,
    PANEL_BG2,
    ACCENT,
    ACCENT2,
    BORDER,
    TEXT,
    TEXT_DIM,
    CANVAS_BG,
    GRID_COL,
    FONT_SECTION,
    FONT_LABEL,
    FONT_SMALL,
    FONT_MONO,
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

        tk.Frame(self, bg=col, width=4).pack(side="left", fill="y", padx=(0, 8))

        tk.Label(
            self,
            text=s.name.upper(),
            bg=PANEL_BG,
            fg=col,
            font=FONT_LABEL,
            width=7,
            anchor="w",
        ).pack(side="left")

        self._mute_var = tk.BooleanVar(value=s.mute)
        self._mute_btn = tk.Checkbutton(
            self,
            text="M",
            variable=self._mute_var,
            bg=PANEL_BG,
            fg=TEXT_DIM,
            selectcolor="#3a1a1a",
            activebackground=PANEL_BG,
            activeforeground=ACCENT,
            font=FONT_SMALL,
            indicatoron=False,
            relief="flat",
            bd=0,
            padx=5,
            command=self._on_mute,
        )
        self._mute_btn.pack(side="left", padx=2)

        self._solo_var = tk.BooleanVar(value=s.solo)
        self._solo_btn = tk.Checkbutton(
            self,
            text="S",
            variable=self._solo_var,
            bg=PANEL_BG,
            fg=TEXT_DIM,
            selectcolor="#1a3a1a",
            activebackground=PANEL_BG,
            activeforeground="#81c784",
            font=FONT_SMALL,
            indicatoron=False,
            relief="flat",
            bd=0,
            padx=5,
            command=self._on_solo,
        )
        self._solo_btn.pack(side="left", padx=2)

        tk.Label(
            self, text="Gain", bg=PANEL_BG, fg=TEXT_DIM, font=FONT_SMALL
        ).pack(side="left", padx=(8, 2))

        self._gain_var = tk.DoubleVar(value=s.gain_db)

        tk.Button(
            self,
            text="◀",
            bg=PANEL_BG2,
            fg=TEXT,
            relief="flat",
            bd=0,
            font=FONT_SMALL,
            padx=4,
            command=lambda: self._step_gain(-0.5),
        ).pack(side="left", padx=(0, 2))

        self._gain_sl = tk.Scale(
            self,
            variable=self._gain_var,
            from_=-24,
            to=6,
            resolution=0.5,
            orient="horizontal",
            length=75,
            bg=PANEL_BG,
            fg=TEXT,
            troughcolor=ACCENT2,
            highlightthickness=0,
            showvalue=False,
            command=self._on_gain,
        )
        self._gain_sl.pack(side="left")

        tk.Button(
            self,
            text="▶",
            bg=PANEL_BG2,
            fg=TEXT,
            relief="flat",
            bd=0,
            font=FONT_SMALL,
            padx=4,
            command=lambda: self._step_gain(+0.5),
        ).pack(side="left", padx=(2, 4))

        self._gain_lbl = tk.Label(
            self,
            text=f"{s.gain_db:+.1f} dB",
            bg=PANEL_BG,
            fg=TEXT,
            font=FONT_MONO,
            width=8,
        )
        self._gain_lbl.pack(side="left")

        tk.Label(
            self, text="Az", bg=PANEL_BG, fg=TEXT_DIM, font=FONT_SMALL
        ).pack(side="left", padx=(8, 2))
        self._az_lbl = tk.Label(
            self,
            text=f"{s.azimuth:+.0f}°",
            bg=PANEL_BG,
            fg=col,
            font=FONT_MONO,
            width=6,
        )
        self._az_lbl.pack(side="left")

        tk.Label(
            self, text="El", bg=PANEL_BG, fg=TEXT_DIM, font=FONT_SMALL
        ).pack(side="left", padx=(8, 2))
        self._el_var = tk.DoubleVar(value=s.elevation)
        self._el_sl = tk.Scale(
            self,
            variable=self._el_var,
            from_=-30,
            to=90,
            resolution=1,
            orient="horizontal",
            length=70,
            bg=PANEL_BG,
            fg=TEXT,
            troughcolor=ACCENT2,
            highlightthickness=0,
            showvalue=False,
            command=self._on_elevation,
        )
        self._el_sl.pack(side="left")

        self._el_lbl = tk.Label(
            self,
            text=f"{s.elevation:+.0f}°",
            bg=PANEL_BG,
            fg=TEXT,
            font=FONT_MONO,
            width=5,
        )
        self._el_lbl.pack(side="left", padx=(4, 0))

        tk.Button(
            self,
            text="…",
            bg=ACCENT2,
            fg=TEXT,
            relief="flat",
            bd=0,
            font=FONT_SMALL,
            padx=6,
            command=self._pick_wav,
        ).pack(side="left", padx=(8, 2))

        self._file_lbl = tk.Label(
            self,
            text="no file",
            bg=PANEL_BG,
            fg=TEXT_DIM,
            font=FONT_SMALL,
            width=14,
            anchor="w",
        )
        self._file_lbl.pack(side="left", padx=2)

    def _step_gain(self, delta):
        v = max(-24, min(6, self._gain_var.get() + delta))
        self._gain_var.set(v)
        self._on_gain()

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
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
        )
        if p:
            self.source.wav_path = p
            self._file_lbl.config(text=Path(p).name[:14])

    def refresh_az(self):
        self._az_lbl.config(text=f"{self.source.azimuth:+.0f}°")


class SceneView(tk.Canvas):
    RADIUS = 175
    NODE_R = 11

    def __init__(self, parent, state: AppState, **kwargs):
        super().__init__(parent, bg=CANVAS_BG, highlightthickness=0, **kwargs)
        self.state = state
        self._rows = []
        self._drag = None

        self.bind("<Configure>", lambda _: self.redraw())
        self.bind("<Button-1>", self._on_press)
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

        w = self.winfo_width()
        h = self.winfo_height()
        if w < 2 or h < 2:
            return

        cx, cy = self._center()
        R = self.RADIUS

        steps = 90
        for i in range(steps):
            t = i / max(1, steps - 1)
            r = int(8 + (18 - 8) * t)
            g = int(12 + (20 - 12) * t)
            b = int(24 + (36 - 24) * t)
            col = f"#{r:02x}{g:02x}{b:02x}"
            y1 = int(i * h / steps)
            y2 = int((i + 1) * h / steps)
            self.create_rectangle(0, y1, w, y2, outline="", fill=col)

        glow_layers = [
            (150, "#0e1930"),
            (115, "#12213d"),
            (85, "#15294a"),
            (55, "#183153"),
        ]
        for r, col in glow_layers:
            self.create_oval(cx - r, cy - r, cx + r, cy + r, outline="", fill=col)

        self.create_rectangle(1, 1, w - 1, h - 1, outline="#1e2c47", width=1)

        ring_specs = [
            (0.33, "#1a2b46", 1),
            (0.66, "#213654", 1),
            (1.00, "#2b4a73", 1),
        ]
        for frac, col, width in ring_specs:
            r = R * frac
            self.create_oval(cx - r, cy - r, cx + r, cy + r, outline=col, width=width)

        self.create_line(
            cx - R - 8, cy, cx + R + 8, cy,
            fill="#22395a", width=1, dash=(4, 4)
        )
        self.create_line(
            cx, cy - R - 8, cx, cy + R + 8,
            fill="#22395a", width=1, dash=(4, 4)
        )

        label_col = "#8ea0c2"
        self.create_text(cx, cy - R - 40, text="FRONT", fill=label_col, font=("Helvetica", 9, "bold"))
        self.create_text(cx, cy + R + 40, text="BACK", fill=label_col, font=("Helvetica", 9, "bold"))
        self.create_text(cx - R - 40, cy, text="LEFT", fill=label_col, font=("Helvetica", 9, "bold"))
        self.create_text(cx + R + 40, cy, text="RIGHT", fill=label_col, font=("Helvetica", 9, "bold"))

        self.create_oval(cx - 16, cy - 16, cx + 16, cy + 16,
                         outline="#203a61", width=1, fill="#0f1728")
        self.create_oval(cx - 8, cy - 8, cx + 8, cy + 8,
                         fill="#101d34", outline=ACCENT, width=2)
        self.create_text(cx, cy, text="•", fill=ACCENT, font=("Helvetica", 15, "bold"))

        for i, src in enumerate(self.state.sources):
            if src.mute:
                continue

            x, y = self._source_to_xy(src)
            nr = self.NODE_R
            col = src.color

            self.create_line(
                cx, cy, x, y,
                fill=col, width=1, dash=(3, 3), tags=f"stem_{i}"
            )

            self.create_oval(
                x - (nr + 7), y - (nr + 7),
                x + (nr + 7), y + (nr + 7),
                outline="", fill="#17233a"
            )

            el_r = nr + 4 + (src.elevation / 90) * 12
            self.create_oval(
                x - el_r, y - el_r, x + el_r, y + el_r,
                outline=col, width=1, dash=(2, 3), tags=f"elring_{i}"
            )

            self.create_oval(
                x - nr, y - nr, x + nr, y + nr,
                fill=col, outline="white", width=1.5,
                tags=(f"node_{i}", "node")
            )

            self.create_oval(
                x - nr + 3, y - nr + 3, x - nr + 7, y - nr + 7,
                outline="", fill="#ffffff"
            )

            self.create_text(
                x, y - nr - 12,
                text=src.name,
                fill=col,
                font=("Helvetica", 8, "bold"),
                tags=f"lbl_{i}"
            )

            self.create_text(
                x, y + nr + 12,
                text=f"{src.azimuth:+.0f}°",
                fill="#aab7d1",
                font=("Courier", 8)
            )

    def _on_press(self, event):
        nr = self.NODE_R + 6
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
            tk.Label(
                self,
                text=text,
                bg=PANEL_BG,
                fg=ACCENT,
                font=FONT_SECTION,
            ).pack(anchor="w", pady=(12, 4), padx=12)

        def dim(text):
            tk.Label(
                self,
                text=text,
                bg=PANEL_BG,
                fg=TEXT_DIM,
                font=FONT_SMALL,
                wraplength=260,
                justify="left",
            ).pack(anchor="w", padx=12)

        section("SONG INPUT")
        frm = tk.Frame(self, bg=PANEL_BG)
        frm.pack(fill="x", padx=12)
        self._song_lbl = tk.Label(
            frm,
            text="no file selected",
            bg=PANEL_BG,
            fg=TEXT_DIM,
            font=FONT_SMALL,
            anchor="w",
        )
        self._song_lbl.pack(side="left", fill="x", expand=True)
        tk.Button(
            frm,
            text="Browse…",
            bg=ACCENT2,
            fg=TEXT,
            relief="flat",
            bd=0,
            font=FONT_SMALL,
            command=self._pick_song,
        ).pack(side="right")
        dim("Leave blank if you load individual stem WAVs in the Sources panel.")

        section("RENDERER")
        self._renderer_var = tk.StringVar(value=s.renderer)
        renderers = [
            ("binaural", "Binaural (HOA → HRTF)"),
            ("ls17_binaural", "LS17 → Binaural"),
            ("ls17", "LS17 decoded (17-channel WAV)"),
        ]
        for val, label in renderers:
            tk.Radiobutton(
                self,
                text=label,
                variable=self._renderer_var,
                value=val,
                bg=PANEL_BG,
                fg=TEXT,
                selectcolor=ACCENT2,
                activebackground=PANEL_BG,
                activeforeground=TEXT,
                font=FONT_SMALL,
                command=self._on_renderer,
            ).pack(anchor="w", padx=16)

        section("SPEAKER LAYOUT")
        frm2 = tk.Frame(self, bg=PANEL_BG)
        frm2.pack(fill="x", padx=12)
        self._layout_lbl = tk.Label(
            frm2,
            text="default (museum 17ch)",
            bg=PANEL_BG,
            fg=TEXT_DIM,
            font=FONT_SMALL,
            anchor="w",
        )
        self._layout_lbl.pack(side="left", fill="x", expand=True)
        tk.Button(
            frm2,
            text="Load CSV…",
            bg=ACCENT2,
            fg=TEXT,
            relief="flat",
            bd=0,
            font=FONT_SMALL,
            command=self._pick_layout,
        ).pack(side="right")

        section("HRTF")
        frm3 = tk.Frame(self, bg=PANEL_BG)
        frm3.pack(fill="x", padx=12)
        self._hrtf_lbl = tk.Label(
            frm3,
            text="default HRTF",
            bg=PANEL_BG,
            fg=TEXT_DIM,
            font=FONT_SMALL,
            anchor="w",
        )
        self._hrtf_lbl.pack(side="left", fill="x", expand=True)
        tk.Button(
            frm3,
            text="Load SOFA…",
            bg=ACCENT2,
            fg=TEXT,
            relief="flat",
            bd=0,
            font=FONT_SMALL,
            command=self._pick_hrtf,
        ).pack(side="right")

        section("HOA ORDER")
        frm4 = tk.Frame(self, bg=PANEL_BG)
        frm4.pack(fill="x", padx=12)
        tk.Label(frm4, text="Order:", bg=PANEL_BG, fg=TEXT, font=FONT_LABEL).pack(side="left")
        self._order_var = tk.IntVar(value=s.hoa_order)
        for o in (1, 2, 3):
            tk.Radiobutton(
                frm4,
                text=str(o),
                variable=self._order_var,
                value=o,
                bg=PANEL_BG,
                fg=TEXT,
                selectcolor=ACCENT2,
                activebackground=PANEL_BG,
                font=FONT_SMALL,
                command=lambda: setattr(s, "hoa_order", self._order_var.get()),
            ).pack(side="left", padx=6)

        section("OUTPUT DIRECTORY")
        frm5 = tk.Frame(self, bg=PANEL_BG)
        frm5.pack(fill="x", padx=12, pady=(0, 10))
        self._outdir_lbl = tk.Label(
            frm5,
            text=str(s.out_dir),
            bg=PANEL_BG,
            fg=TEXT_DIM,
            font=FONT_SMALL,
            anchor="w",
        )
        self._outdir_lbl.pack(side="left", fill="x", expand=True)
        tk.Button(
            frm5,
            text="Change…",
            bg=ACCENT2,
            fg=TEXT,
            relief="flat",
            bd=0,
            font=FONT_SMALL,
            command=self._pick_outdir,
        ).pack(side="right")

    def _pick_song(self):
        p = filedialog.askopenfilename(
            title="Select song file",
            filetypes=[("Audio files", "*.wav *.mp3 *.flac"), ("All files", "*.*")],
        )
        if p:
            self.state.song_path = p
            self._song_lbl.config(text=Path(p).name)

    def _on_renderer(self):
        self.state.renderer = self._renderer_var.get()

    def _pick_layout(self):
        p = filedialog.askopenfilename(
            title="Select speaker layout",
            filetypes=[("CSV / JSON", "*.csv *.json"), ("All files", "*.*")],
        )
        if p:
            self.state.layout_path = p
            self._layout_lbl.config(text=Path(p).name)

    def _pick_hrtf(self):
        p = filedialog.askopenfilename(
            title="Select HRTF SOFA file",
            filetypes=[("SOFA files", "*.sofa"), ("All files", "*.*")],
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
        super().__init__(parent, bg="#0d1117", height=24, **kwargs)
        self._var = tk.StringVar(value="Ready.")
        tk.Label(
            self,
            textvariable=self._var,
            bg="#0d1117",
            fg=TEXT_DIM,
            font=FONT_SMALL,
            anchor="w",
        ).pack(side="left", padx=10)

    def set(self, msg: str):
        self._var.set(msg)
        self.update_idletasks()