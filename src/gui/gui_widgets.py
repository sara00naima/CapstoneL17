import math
import threading
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
    FONT_SECTION,
    FONT_LABEL,
    FONT_SMALL,
    FONT_MONO,
    run_demix_and_populate,
)


class SourceRow(tk.Frame):
    def __init__(self, parent, source: SourceState, scene_view, on_select=None, **kwargs):
        super().__init__(parent, bg=PANEL_BG, height=34, **kwargs)
        self.source = source
        self.scene_view = scene_view
        self.on_select = on_select
        self.pack_propagate(False)
        self._build()

    def _build(self):
        s = self.source
        col = s.color

        tk.Frame(self, bg=col, width=4).pack(side="left", fill="y", padx=(0, 8))

        self._name = tk.Label(
            self,
            text=s.name.upper(),
            bg=PANEL_BG,
            fg=col,
            font=("Helvetica", 10, "bold"),
            anchor="w",
        )
        self._name.pack(side="left", fill="x", expand=True)

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
            font=("Helvetica", 8, "bold"),
            indicatoron=False,
            relief="flat",
            bd=0,
            padx=4,
            command=self._on_solo,
        )
        self._solo_btn.pack(side="right", padx=(2, 2))

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
            font=("Helvetica", 8, "bold"),
            indicatoron=False,
            relief="flat",
            bd=0,
            padx=4,
            command=self._on_mute,
        )
        self._mute_btn.pack(side="right", padx=(2, 8))

        self._az_lbl = tk.Label(
            self,
            text=f"{s.azimuth:+.0f}°",
            bg=PANEL_BG,
            fg=TEXT_DIM,
            font=("Courier", 9),
            width=6,
            anchor="e",
        )
        self._az_lbl.pack(side="right", padx=(6, 2))

        for widget in (self, self._name, self._az_lbl):
            widget.bind("<Button-1>", self._select)

    def _select(self, _event=None):
        if self.on_select:
            self.on_select(self.source)

    def _on_mute(self):
        self.source.mute = self._mute_var.get()
        self._mute_btn.config(fg=ACCENT if self.source.mute else TEXT_DIM)
        self.scene_view.redraw()

    def _on_solo(self):
        self.source.solo = self._solo_var.get()
        self._solo_btn.config(fg="#81c784" if self.source.solo else TEXT_DIM)

    def refresh_az(self):
        self._az_lbl.config(text=f"{self.source.azimuth:+.0f}°")

    def refresh_all(self):
        """Called after demix to update azimuth label. File is shown in inspector."""
        self.refresh_az()


class SourceInspector(tk.Frame):
    def __init__(self, parent, scene_view, **kwargs):
        super().__init__(parent, bg=PANEL_BG, **kwargs)
        self.scene_view = scene_view
        self.source = None
        self._build()

    def _build(self):
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", pady=(4, 10))

        tk.Label(
            self,
            text="SELECTED SOURCE",
            bg=PANEL_BG,
            fg=ACCENT,
            font=FONT_SECTION,
        ).pack(anchor="w", padx=12, pady=(0, 6))

        self._title = tk.Label(
            self,
            text="None",
            bg=PANEL_BG,
            fg=TEXT,
            font=("Helvetica", 11, "bold"),
        )
        self._title.pack(anchor="w", padx=12)

        # Gain
        gain_block = tk.Frame(self, bg=PANEL_BG)
        gain_block.pack(fill="x", padx=12, pady=(10, 4))

        tk.Label(
            gain_block, text="Gain", bg=PANEL_BG, fg=TEXT_DIM, font=FONT_SMALL
        ).pack(anchor="w")

        gain_line = tk.Frame(gain_block, bg=PANEL_BG)
        gain_line.pack(fill="x", pady=(3, 0))

        self._gain_var = tk.DoubleVar(value=0)

        tk.Button(
            gain_line,
            text="◀",
            bg=PANEL_BG2,
            fg=TEXT,
            relief="flat",
            bd=0,
            font=FONT_SMALL,
            command=lambda: self._step_gain(-0.5),
        ).pack(side="left", padx=(0, 2))

        self._gain_sl = tk.Scale(
            gain_line,
            variable=self._gain_var,
            from_=-24,
            to=6,
            resolution=0.5,
            orient="horizontal",
            length=180,
            bg=PANEL_BG,
            fg=TEXT,
            troughcolor=ACCENT,
            highlightthickness=0,
            showvalue=False,
            command=self._on_gain,
        )
        self._gain_sl.pack(side="left")

        tk.Button(
            gain_line,
            text="▶",
            bg=PANEL_BG2,
            fg=TEXT,
            relief="flat",
            bd=0,
            font=FONT_SMALL,
            command=lambda: self._step_gain(+0.5),
        ).pack(side="left", padx=(2, 8))

        self._gain_lbl = tk.Label(
            gain_line,
            text="+0.0 dB",
            bg=PANEL_BG,
            fg=TEXT,
            font=FONT_MONO,
            width=8,
            anchor="w",
        )
        self._gain_lbl.pack(side="left")

        # Azimuth
        az_block = tk.Frame(self, bg=PANEL_BG)
        az_block.pack(fill="x", padx=12, pady=(8, 2))

        tk.Label(
            az_block, text="Azimuth", bg=PANEL_BG, fg=TEXT_DIM, font=FONT_SMALL
        ).pack(anchor="w")

        self._az_lbl = tk.Label(
            az_block,
            text="+0°",
            bg=PANEL_BG,
            fg=TEXT,
            font=FONT_MONO,
            anchor="w",
        )
        self._az_lbl.pack(anchor="w", pady=(3, 0))

        # Elevation
        el_block = tk.Frame(self, bg=PANEL_BG)
        el_block.pack(fill="x", padx=12, pady=(8, 2))

        tk.Label(
            el_block, text="Elevation", bg=PANEL_BG, fg=TEXT_DIM, font=FONT_SMALL
        ).pack(anchor="w")

        el_line = tk.Frame(el_block, bg=PANEL_BG)
        el_line.pack(fill="x", pady=(3, 0))

        self._el_var = tk.DoubleVar(value=0)

        self._el_sl = tk.Scale(
            el_line,
            variable=self._el_var,
            from_=-30,
            to=90,
            resolution=1,
            orient="horizontal",
            length=190,
            bg=PANEL_BG,
            fg=TEXT,
            troughcolor=ACCENT,
            highlightthickness=0,
            showvalue=False,
            command=self._on_elevation,
        )
        self._el_sl.pack(side="left")

        self._el_lbl = tk.Label(
            el_line,
            text="+0°",
            bg=PANEL_BG,
            fg=TEXT,
            font=FONT_MONO,
            width=6,
        )
        self._el_lbl.pack(side="left", padx=(8, 0))

        # File
        file_block = tk.Frame(self, bg=PANEL_BG)
        file_block.pack(fill="x", padx=12, pady=(8, 12))

        tk.Label(
            file_block, text="File", bg=PANEL_BG, fg=TEXT_DIM, font=FONT_SMALL
        ).pack(anchor="w")

        self._file_lbl = tk.Label(
            file_block,
            text="no file",
            bg=PANEL_BG,
            fg=TEXT,
            font=FONT_SMALL,
            anchor="w",
            justify="left",
            wraplength=250,
        )
        self._file_lbl.pack(anchor="w", pady=(3, 6))

        tk.Button(
            file_block,
            text="Load WAV…",
            bg=ACCENT,
            fg=TEXT,
            relief="flat",
            bd=0,
            font=FONT_SMALL,
            command=self._pick_wav,
        ).pack(anchor="w")

    def set_source(self, source: SourceState):
        self.source = source
        self._title.config(text=source.name.upper(), fg=source.color)
        self._gain_var.set(source.gain_db)
        self._gain_lbl.config(text=f"{source.gain_db:+.1f} dB")
        self._az_lbl.config(text=f"{source.azimuth:+.0f}°")
        self._el_var.set(source.elevation)
        self._el_lbl.config(text=f"{source.elevation:+.0f}°")
        self._file_lbl.config(text=Path(source.wav_path).name if source.wav_path else "no file")

    def update_azimuth(self):
        if self.source:
            self._az_lbl.config(text=f"{self.source.azimuth:+.0f}°")

    def _step_gain(self, delta):
        if not self.source:
            return
        v = max(-24, min(6, self._gain_var.get() + delta))
        self._gain_var.set(v)
        self._on_gain()

    def _on_gain(self, _=None):
        if not self.source:
            return
        v = self._gain_var.get()
        self.source.gain_db = v
        self._gain_lbl.config(text=f"{v:+.1f} dB")

    def _on_elevation(self, _=None):
        if not self.source:
            return
        v = self._el_var.get()
        self.source.elevation = v
        self._el_lbl.config(text=f"{v:+.0f}°")
        self.scene_view.redraw()

    def _pick_wav(self):
        if not self.source:
            return
        p = filedialog.askopenfilename(
            title=f"Select WAV for {self.source.name}",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
        )
        if p:
            self.source.wav_path = p
            self._file_lbl.config(text=Path(p).name)


class SceneView(tk.Canvas):
    NODE_R = 12

    def __init__(self, parent, state: AppState, **kwargs):
        super().__init__(parent, bg=CANVAS_BG, highlightthickness=0, **kwargs)
        self.state = state
        self._rows = []
        self._inspector = None
        self._drag = None

        self.bind("<Configure>", lambda _e: self.redraw())
        self.bind("<Button-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)

    def set_rows(self, rows):
        self._rows = rows

    def set_inspector(self, inspector):
        self._inspector = inspector

    def _center(self):
        return self.winfo_width() / 2, self.winfo_height() / 2

    def _effective_radius(self):
        w = self.winfo_width()
        h = self.winfo_height()
        return max(120, min(w, h) * 0.34)

    def _source_to_xy(self, source: SourceState):
        cx, cy = self._center()
        r = self._effective_radius() * math.cos(math.radians(max(0, source.elevation)))
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
        R = self._effective_radius()

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

        for rr, col in [
            (R * 0.82, "#0e1930"),
            (R * 0.62, "#12213d"),
            (R * 0.44, "#15294a"),
            (R * 0.28, "#183153"),
        ]:
            self.create_oval(cx - rr, cy - rr, cx + rr, cy + rr, outline="", fill=col)

        self.create_rectangle(1, 1, w - 1, h - 1, outline="#1e2c47", width=1)

        for frac, col, width in [
            (0.33, "#1a2b46", 1),
            (0.66, "#213654", 1),
            (1.00, "#2b4a73", 1),
        ]:
            rr = R * frac
            self.create_oval(cx - rr, cy - rr, cx + rr, cy + rr, outline=col, width=width)

        self.create_line(cx - R - 10, cy, cx + R + 10, cy, fill="#22395a", width=1, dash=(4, 4))
        self.create_line(cx, cy - R - 10, cx, cy + R + 10, fill="#22395a", width=1, dash=(4, 4))

        label_col = "#7f93b8"
        self.create_text(cx, cy - R - 40, text="FRONT", fill=label_col, font=("Helvetica", 9, "bold"), anchor="s")
        self.create_text(cx, cy + R + 40, text="BACK", fill=label_col, font=("Helvetica", 9, "bold"), anchor="n")
        self.create_text(cx - R - 40, cy, text="LEFT", fill=label_col, font=("Helvetica", 9, "bold"), anchor="e")
        self.create_text(cx + R + 40, cy, text="RIGHT", fill=label_col, font=("Helvetica", 9, "bold"), anchor="w")

        self.create_oval(cx - 15, cy - 15, cx + 15, cy + 15, outline="#203a61", width=1, fill="#0f1728")
        self.create_oval(cx - 8, cy - 8, cx + 8, cy + 8, fill="#101d34", outline=ACCENT, width=2)
        self.create_text(cx, cy, text="•", fill=ACCENT, font=("Helvetica", 14, "bold"))

        for i, src in enumerate(self.state.sources):
            if src.mute:
                continue

            x, y = self._source_to_xy(src)
            nr = self.NODE_R
            col = src.color

            self.create_line(cx, cy, x, y, fill=col, width=1, dash=(3, 3))
            self.create_oval(x - (nr + 7), y - (nr + 7), x + (nr + 7), y + (nr + 7), outline="", fill="#17233a")

            el_r = nr + 5 + (src.elevation / 90) * 11
            self.create_oval(x - el_r, y - el_r, x + el_r, y + el_r, outline=col, width=1, dash=(2, 3))

            self.create_oval(x - nr, y - nr, x + nr, y + nr, fill=col, outline="white", width=1.5)
            self.create_oval(x - nr + 3, y - nr + 3, x - nr + 6, y - nr + 6, outline="", fill="#ffffff")

            self.create_text(
                x,
                y - nr - 10,
                text=src.name,
                fill="#c8d4ea",
                font=("Helvetica", 8, "bold"),
                anchor="s",
            )

            self.create_text(
                x,
                y + nr + 10,
                text=f"{src.azimuth:+.0f}°",
                fill="#aab7d1",
                font=("Courier", 8),
                anchor="n",
            )

    def _on_press(self, event):
        nr = self.NODE_R + 8
        for i, src in enumerate(self.state.sources):
            if src.mute:
                continue
            x, y = self._source_to_xy(src)
            if abs(event.x - x) < nr and abs(event.y - y) < nr:
                self._drag = i
                if self._inspector is not None:
                    self._inspector.set_source(src)
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

        if self._inspector is not None and self._inspector.source is src:
            self._inspector.update_azimuth()

    def _on_release(self, _event):
        self._drag = None


class OutputPanel(tk.Frame):
    def __init__(self, parent, state: AppState,
                 status_ref=None, scene_ref=None,
                 inspector_ref=None, rows_ref=None, **kwargs):
        super().__init__(parent, bg=PANEL_BG, **kwargs)
        self.state = state
        self._status_ref    = status_ref
        self._scene_ref     = scene_ref
        self._inspector_ref = inspector_ref
        self._rows_ref      = rows_ref or []
        self._build()

    def _build(self):
        s = self.state

        def section(text):
            tk.Label(
                self,
                text=text,
                bg=PANEL_BG,
                fg=ACCENT,
                font=("Helvetica", 11, "bold"),
            ).pack(anchor="w", pady=(12, 4), padx=12)

        def small_value(text=""):
            return tk.Label(
                self,
                text=text,
                bg=PANEL_BG,
                fg=TEXT_DIM,
                font=FONT_SMALL,
                anchor="w",
                justify="left",
                wraplength=250,
            )

        section("SONG & DEMIX")

        # Song file row
        song_row = tk.Frame(self, bg=PANEL_BG)
        song_row.pack(fill="x", padx=12, pady=(0, 4))
        self._song_lbl = tk.Label(
            song_row, text="no song selected", bg=PANEL_BG, fg=TEXT_DIM,
            font=FONT_SMALL, anchor="w", justify="left", wraplength=170,
        )
        self._song_lbl.pack(side="left", fill="x", expand=True)
        tk.Button(
            song_row, text="Browse…", bg=ACCENT2, fg=TEXT,
            relief="flat", bd=0, font=FONT_SMALL, command=self._pick_song,
        ).pack(side="right")

        # Model checkpoint row
        model_row = tk.Frame(self, bg=PANEL_BG)
        model_row.pack(fill="x", padx=12, pady=(0, 4))
        self._model_lbl = tk.Label(
            model_row, text="no model (.ckpt)", bg=PANEL_BG, fg=TEXT_DIM,
            font=FONT_SMALL, anchor="w", justify="left", wraplength=170,
        )
        self._model_lbl.pack(side="left", fill="x", expand=True)
        tk.Button(
            model_row, text="Model…", bg=ACCENT2, fg=TEXT,
            relief="flat", bd=0, font=FONT_SMALL, command=self._pick_model,
        ).pack(side="right")

        # Demix button
        self._demix_btn = tk.Button(
            self, text="🎵 Demix Song",
            bg="#1e3a5f", fg=TEXT,
            activebackground="#2a4f7a", activeforeground=TEXT,
            relief="flat", bd=0,
            font=("Helvetica", 10, "bold"),
            padx=12, pady=6, cursor="hand2",
            command=self._on_demix,
        )
        self._demix_btn.pack(anchor="w", padx=12, pady=(4, 2))

        tk.Label(
            self,
            text="Or load individual stem WAVs via Sources inspector → Load WAV…",
            bg=PANEL_BG, fg=TEXT_DIM, font=FONT_SMALL,
            justify="left", wraplength=250,
        ).pack(anchor="w", padx=12, pady=(2, 6))

        section("RENDERER")
        self._renderer_var = tk.StringVar(value=s.renderer)
        for val, label in [
            ("binaural", "Binaural (HOA → HRTF)"),
            ("ls17_binaural", "LS17 → Binaural"),
            ("ls17", "LS17 decoded (17-channel WAV)"),
        ]:
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
                anchor="w",
                justify="left",
                wraplength=250,
                command=self._on_renderer,
            ).pack(anchor="w", padx=14, pady=1)

        section("SPEAKER LAYOUT")
        self._layout_lbl = small_value("default (museum 17ch)")
        self._layout_lbl.pack(anchor="w", padx=12)
        tk.Button(
            self,
            text="Load CSV…",
            bg=ACCENT2,
            fg=TEXT,
            relief="flat",
            bd=0,
            font=FONT_SMALL,
            command=self._pick_layout,
        ).pack(anchor="w", padx=12, pady=(6, 0))

        section("HRTF")
        self._hrtf_lbl = small_value("default HRTF")
        self._hrtf_lbl.pack(anchor="w", padx=12)
        tk.Button(
            self,
            text="Load SOFA…",
            bg=ACCENT2,
            fg=TEXT,
            relief="flat",
            bd=0,
            font=FONT_SMALL,
            command=self._pick_hrtf,
        ).pack(anchor="w", padx=12, pady=(6, 0))

        section("HOA ORDER")
        order_row = tk.Frame(self, bg=PANEL_BG)
        order_row.pack(fill="x", padx=12)
        tk.Label(
            order_row,
            text="Order:",
            bg=PANEL_BG,
            fg=TEXT,
            font=FONT_LABEL,
        ).pack(side="left")

        self._order_var = tk.IntVar(value=s.hoa_order)
        for o in (1, 2, 3):
            tk.Radiobutton(
                order_row,
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
        self._outdir_lbl = small_value(str(s.out_dir))
        self._outdir_lbl.pack(anchor="w", padx=12)
        tk.Button(
            self,
            text="Change…",
            bg=ACCENT2,
            fg=TEXT,
            relief="flat",
            bd=0,
            font=FONT_SMALL,
            command=self._pick_outdir,
        ).pack(anchor="w", padx=12, pady=(6, 0))

    def _pick_song(self):
        p = filedialog.askopenfilename(
            title="Select song file",
            filetypes=[("Audio files", "*.wav *.mp3 *.flac"), ("All files", "*.*")],
        )
        if p:
            self.state.song_path = p
            self._song_lbl.config(text=Path(p).name)

    def _pick_model(self):
        p = filedialog.askopenfilename(
            title="Select BS-RoFormer model checkpoint",
            filetypes=[("Checkpoint", "*.ckpt *.pt *.pth"), ("All files", "*.*")],
        )
        if p:
            self.state.demix_model_path = p
            self._model_lbl.config(text=Path(p).name)

    def _on_demix(self):
        t = threading.Thread(
            target=run_demix_and_populate,
            args=(self.state, self._status_ref, self._demix_btn, self._after_demix),
            daemon=True,
        )
        t.start()

    def _after_demix(self):
        """Runs on the main thread after demix completes. Refreshes all widgets."""
        # 1. Refresh azimuth labels on every source row
        for row in self._rows_ref:
            row.refresh_all()

        # 2. Reload the inspector so it shows the new file name, az, el
        if self._inspector_ref is not None and self._inspector_ref.source is not None:
            self._inspector_ref.set_source(self._inspector_ref.source)

        # 3. Redraw the scene view (nodes may have moved)
        if self._scene_ref is not None:
            self._scene_ref.redraw()

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
        super().__init__(parent, bg="#0d1117", height=26, **kwargs)
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