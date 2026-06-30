import math
import time
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

from PIL import Image, ImageDraw, ImageTk


from gui_backend import (
    BG_2,
    AppState,
    SourceState,
    PANEL_BG,
    PANEL_BG2,
    ROW_SELECTED,
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
    ICONS_DIR,
    run_demix_and_populate,
    populate_sources_from_stem_paths,
)


def _pil_rgb(hex_color: str) -> tuple[int, int, int]:
    c = hex_color.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def _pil_dashed_line(draw, x1, y1, x2, y2, fill, width, dash):
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 1:
        return
    dx, dy = (x2 - x1) / length, (y2 - y1) / length
    on, budget, pos = True, dash[0], 0.0
    while pos < length:
        end = min(pos + budget, length)
        if on:
            draw.line(
                [(x1 + dx * pos, y1 + dy * pos), (x1 + dx * end, y1 + dy * end)],
                fill=fill, width=width,
            )
        budget -= end - pos
        if budget <= 0:
            on = not on
            budget = dash[0] if on else dash[1]
        pos = end


def _pil_dashed_ellipse(draw, x1, y1, x2, y2, fill, width, dash):
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    rx, ry = (x2 - x1) / 2, (y2 - y1) / 2
    if rx < 1 or ry < 1:
        return
    n = 120
    pts = [
        (cx + rx * math.cos(2 * math.pi * i / n),
         cy + ry * math.sin(2 * math.pi * i / n))
        for i in range(n + 1)
    ]
    on, budget = True, dash[0]
    for i in range(n):
        px1, py1 = pts[i]
        px2, py2 = pts[i + 1]
        seg = math.hypot(px2 - px1, py2 - py1)
        if seg < 0.001:
            continue
        dx, dy = (px2 - px1) / seg, (py2 - py1) / seg
        pos = 0.0
        while pos < seg:
            end = min(pos + budget, seg)
            if on:
                draw.line(
                    [(px1 + dx * pos, py1 + dy * pos),
                     (px1 + dx * end, py1 + dy * end)],
                    fill=fill, width=width,
                )
            budget -= end - pos
            if budget <= 0:
                on = not on
                budget = dash[0] if on else dash[1]
            pos = end


def _load_svg_icon(svg_path: str, size: int, color: tuple) -> Image.Image:
    """Rasterize a simple SVG (M/L/H/V/Z paths + circles, currentColor) to PIL RGBA.
    Renders at 4× and downscales with LANCZOS for smooth anti-aliased edges."""
    import xml.etree.ElementTree as ET
    import re

    SS = 4  # supersampling factor
    rs = size * SS  # render size

    tree = ET.parse(svg_path)
    root = tree.getroot()
    ns_prefix = root.tag.split("}")[0] + "}" if "}" in root.tag else ""

    scale = rs / 24.0
    root_fill   = root.get("fill",         "none")
    root_stroke = root.get("stroke",       "none")
    root_sw     = float(root.get("stroke-width", "1.0"))

    def _resolve(elem, attr, default):
        v = elem.get(attr, default)
        return color if v == "currentColor" else None

    def _sw(elem):
        return max(1, round(float(elem.get("stroke-width", root_sw)) * scale))

    def _parse_path(d: str):
        tokens = re.findall(r'[MLHVZmlhvz]|[-+]?\d*\.?\d+', d)
        polylines, current = [], []
        x, y, cmd = 0.0, 0.0, None
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t in "MLHVZmlhvz":
                cmd = t
                i += 1
                if cmd in "Zz" and current:
                    current.append(current[0])
                    polylines.append(current)
                    current = []
            elif cmd == "M":
                x, y = float(tokens[i]), float(tokens[i + 1])
                if current:
                    polylines.append(current)
                current = [(x * scale, y * scale)]
                i += 2
            elif cmd == "L":
                x, y = float(tokens[i]), float(tokens[i + 1])
                current.append((x * scale, y * scale))
                i += 2
            elif cmd == "H":
                x = float(tokens[i])
                current.append((x * scale, y * scale))
                i += 1
            elif cmd == "V":
                y = float(tokens[i])
                current.append((x * scale, y * scale))
                i += 1
            else:
                i += 1
        if current:
            polylines.append(current)
        return polylines

    img = Image.new("RGBA", (rs, rs), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    for elem in root.iter():
        tag    = elem.tag.replace(ns_prefix, "")
        fill   = _resolve(elem, "fill",   root_fill)
        stroke = _resolve(elem, "stroke", root_stroke)
        sw     = _sw(elem)

        if tag == "circle":
            cx = float(elem.get("cx", 0)) * scale
            cy = float(elem.get("cy", 0)) * scale
            r  = float(elem.get("r",  0)) * scale
            bb = [cx - r, cy - r, cx + r, cy + r]
            if fill:
                draw.ellipse(bb, fill=fill)
            if stroke:
                draw.ellipse(bb, outline=stroke, width=sw)

        elif tag == "path":
            for pts in _parse_path(elem.get("d", "")):
                if len(pts) < 2:
                    continue
                is_closed = pts[0] == pts[-1] and len(pts) >= 4
                if fill and is_closed:
                    draw.polygon(pts, fill=fill)
                if stroke:
                    draw.line(pts, fill=stroke, width=sw)

    return img.resize((size, size), Image.LANCZOS)


def _make_stop_icon(size: int, color: tuple) -> Image.Image:
    """PIL-drawn filled square (stop symbol) at 4× SS, downscaled with LANCZOS."""
    SS = 4
    rs = size * SS
    img = Image.new("RGBA", (rs, rs), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    m = round(rs * 0.22)
    draw.rectangle([m, m, rs - m - 1, rs - m - 1], fill=color)
    return img.resize((size, size), Image.LANCZOS)


def make_button_3d(btn, base_bg, *, fg=TEXT, border=BORDER, active_bg=None, pressed_bg=None):
    active_bg = active_bg or base_bg
    pressed_bg = pressed_bg or active_bg

    btn.configure(
        bg=base_bg,
        fg=fg,
        activebackground=active_bg,
        activeforeground=fg,
        relief="raised",
        bd=1,
        highlightthickness=1,
        highlightbackground=border,
        highlightcolor=border,
        overrelief="ridge",
        padx=8,
        pady=4,
        cursor="hand2",
    )

    def _press(_event):
        btn.configure(relief="sunken", bg=pressed_bg)

    def _release(_event):
        btn.configure(relief="raised", bg=active_bg)

    def _enter(_event):
        btn.configure(bg=active_bg)

    def _leave(_event):
        btn.configure(relief="raised", bg=base_bg)

    btn.bind("<ButtonPress-1>", _press, add="+")
    btn.bind("<ButtonRelease-1>", _release, add="+")
    btn.bind("<Enter>", _enter, add="+")
    btn.bind("<Leave>", _leave, add="+")
    return btn


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

        self._dot = tk.Canvas(self, width=10, height=10, bg=PANEL_BG, highlightthickness=0)
        self._dot.pack(side="left", padx=(10, 8))
        self._dot.create_oval(1, 1, 9, 9, fill=col, outline="")

        self._name = tk.Label(
            self,
            text=s.name.capitalize(),
            bg=PANEL_BG,
            fg=TEXT,
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

        for widget in (self, self._name, self._az_lbl, self._dot):
            widget.bind("<Button-1>", self._select)

        self._bg_widgets = [self, self._name, self._az_lbl, self._dot,
                            self._solo_btn, self._mute_btn]

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
        self.refresh_az()

    def set_selected(self, selected: bool):
        bg = ROW_SELECTED if selected else PANEL_BG
        for w in self._bg_widgets:
            w.configure(bg=bg)
        self._solo_btn.configure(activebackground=bg)
        self._mute_btn.configure(activebackground=bg)


class SourceInspector(tk.Frame):
    def __init__(self, parent, state: AppState, scene_view, rows=None, **kwargs):
        super().__init__(parent, bg=PANEL_BG, **kwargs)
        self.state = state
        self.scene_view = scene_view
        self.rows = rows or []
        self.source = None
        self._build()

    def _build(self):
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", pady=(4, 0))

        hdr = tk.Frame(self, bg=PANEL_BG)
        hdr.pack(fill="x", pady=(0, 6))
        tk.Frame(hdr, bg=ACCENT, width=3).pack(side="left", fill="y")
        tk.Label(
            hdr,
            text="SELECTED SOURCE",
            bg=PANEL_BG,
            fg=TEXT,
            font=("Helvetica", 10, "bold"),
            anchor="w",
            padx=10,
            pady=5,
        ).pack(side="left")

        self._title = tk.Label(
            self,
            text="—",
            bg=PANEL_BG,
            fg=TEXT_DIM,
            font=("Helvetica", 12, "bold"),
        )
        self._title.pack(anchor="w", padx=14)

        gain_block = tk.Frame(self, bg=PANEL_BG)
        gain_block.pack(fill="x", padx=12, pady=(8, 4))

        tk.Label(
            gain_block, text="Gain", bg=PANEL_BG, fg=TEXT_DIM, font=FONT_SMALL
        ).pack(anchor="w")

        gain_line = tk.Frame(gain_block, bg=PANEL_BG)
        gain_line.pack(fill="x", pady=(3, 0))

        self._gain_var = tk.DoubleVar(value=0)

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

        file_block = tk.Frame(self, bg=PANEL_BG)
        file_block.pack(fill="x", padx=12, pady=(8, 8))

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

    def set_source(self, source: SourceState):
        for row in self.rows:
            row.set_selected(row.source is source)
        self.source = source
        self._title.config(text=source.name.capitalize(), fg=source.color)
        self._gain_var.set(source.gain_db)
        self._gain_lbl.config(text=f"{source.gain_db:+.1f} dB")
        self._az_lbl.config(text=f"{source.azimuth:+.0f}°")
        self._el_var.set(source.elevation)
        self._el_lbl.config(text=f"{source.elevation:+.0f}°")
        self._file_lbl.config(text=Path(source.wav_path).name if source.wav_path else "no file")

    def update_azimuth(self):
        if self.source:
            self._az_lbl.config(text=f"{self.source.azimuth:+.0f}°")

    def update_elevation_display(self):
        if self.source:
            self._el_var.set(self.source.elevation)
            self._el_lbl.config(text=f"{self.source.elevation:+.0f}°")

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



class SceneView(tk.Canvas):
    NODE_R = 12

    def __init__(self, parent, state: AppState, **kwargs):
        super().__init__(parent, bg=CANVAS_BG, highlightthickness=0, **kwargs)
        self.state = state
        self._rows = []
        self._inspector = None
        self._drag = None

        # --- Record Movement state ---
        self._recording = False
        self._record_source = None      # SourceState being recorded
        self._record_start_time = None  # time.perf_counter() at REC start
        self._record_samples = []       # [(t_seconds, azimuth_deg, elevation_deg), ...]
        self._record_btn_ref = None     # external "Record Movement" tk.Button, for visual toggling
        self._live_player = None        # LivePlayer reference during playback
        self._bg_pil = None             # cached PIL gradient image at 2× (rebuilt on resize only)
        self._bg_size = (0, 0)
        self._scene_photo = None        # PhotoImage ref kept to prevent GC

        self.bind("<Configure>", lambda _e: self.redraw())
        self.bind("<Button-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)

    def set_rows(self, rows):
        self._rows = rows

    def set_inspector(self, inspector):
        self._inspector = inspector

    def get_selected_source(self):
        return self._inspector.source if self._inspector is not None else None

    def set_record_button(self, btn):
        self._record_btn_ref = btn

    def set_live_player(self, player):
        """Call with a LivePlayer to start animating trajectories, or None to stop."""
        self._live_player = player
        if player is not None:
            self._animate_live()
        else:
            self.redraw()

    def _animate_live(self):
        if self._live_player is None:
            return
        self.redraw()
        self.after(50, self._animate_live)  # ~20 fps

    # --- Record Movement: public API, called by the "Record Movement" button ---

    def is_recording(self):
        return self._recording

    def toggle_recording(self):
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        source = self._inspector.source if self._inspector is not None else None
        if source is None:
            messagebox.showinfo(
                "No source selected",
                "Select a source first (click a node or a row in SOURCES), "
                "then press Record Movement.",
            )
            return

        self._recording = True
        self._record_source = source
        self._record_start_time = time.perf_counter()
        # Seed the recording with the source's current position at t=0, so
        # even a "click somewhere and release immediately" gesture produces
        # a valid two-point movement (start anchor -> clicked point).
        self._record_samples = [(0.0, source.azimuth, source.elevation)]

        if self._record_btn_ref is not None:
            self._record_btn_ref.config(text="Stop Recording", bg="#B33A3A")
            if hasattr(self._record_btn_ref, "_stop_photo"):
                self._record_btn_ref.config(image=self._record_btn_ref._stop_photo)

        self.redraw()

    def _stop_recording(self):
        self._recording = False
        source = self._record_source

        if source is not None and len(self._record_samples) >= 1:
            # Close the loop: append a final sample at the loop's end time,
            # equal to the last recorded position, so generate_from_recording
            # has a well-defined loop_duration even if the user stopped
            # mid-drag rather than exactly on a sample.
            elapsed = time.perf_counter() - self._record_start_time
            last_t, last_az, last_el = self._record_samples[-1]
            if elapsed > last_t:
                self._record_samples.append((elapsed, last_az, last_el))

            source.recorded_movement = list(self._record_samples)

        self._record_source = None
        self._record_start_time = None
        self._record_samples = []

        if self._record_btn_ref is not None:
            self._record_btn_ref.config(text="Record Movement", bg=ACCENT2)
            if hasattr(self._record_btn_ref, "_record_photo"):
                self._record_btn_ref.config(image=self._record_btn_ref._record_photo)

        if self._rows:
            for row in self._rows:
                if row.source is source:
                    row.refresh_all()

        self.redraw()

    def clear_recorded_movement(self, source):
        """Removes a recorded movement from a source, reverting it to a static anchor."""
        source.recorded_movement = None
        self.redraw()

    def _center(self):
        return self.winfo_width() / 2, self.winfo_height() / 2 + 14

    def _effective_radius(self):
        w = self.winfo_width()
        h = self.winfo_height()
        return max(120, min(w, h) * 0.34)

    def _source_to_xy_deg(self, az: float, el: float):
        cx, cy = self._center()
        r = self._effective_radius() * math.cos(math.radians(max(0, el)))
        angle_rad = math.radians(90 - az)
        x = cx + r * math.cos(angle_rad)
        y = cy - r * math.sin(angle_rad)
        return x, y

    def _source_to_xy(self, source: SourceState):
        return self._source_to_xy_deg(source.azimuth, source.elevation)

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

        S = 4  # supersampling scale
        sw, sh = w * S, h * S
        cx, cy = self._center()
        R = self._effective_radius()

        # --- Background gradient (PIL image cached at 2×, rebuilt only on resize) ---
        if self._bg_size != (sw, sh):
            bg = Image.new("RGB", (sw, sh))
            bg_draw = ImageDraw.Draw(bg)
            for sy in range(sh):
                t = sy / max(1, sh - 1)
                rc = int(36 + (58 - 36) * t)
                gc = int(27 + (44 - 27) * t)
                bc = int(21 + (34 - 21) * t)
                bg_draw.line([(0, sy), (sw - 1, sy)], fill=(rc, gc, bc))
            self._bg_pil = bg
            self._bg_size = (sw, sh)

        img = self._bg_pil.copy()
        draw = ImageDraw.Draw(img)

        # helpers that work in canvas coords but draw at 2× into `img`
        def sc(v):
            return v * S

        def _e(x1, y1, x2, y2, *, fill=None, outline=None, width=1):
            kw = {"width": max(1, round(width * S))}
            if fill:
                kw["fill"] = _pil_rgb(fill)
            if outline:
                kw["outline"] = _pil_rgb(outline)
            draw.ellipse([sc(x1), sc(y1), sc(x2), sc(y2)], **kw)

        def _dl(x1, y1, x2, y2, *, fill, width=1, dash=None):
            f = _pil_rgb(fill)
            lw = max(1, round(width * S))
            if dash is None:
                draw.line([(sc(x1), sc(y1)), (sc(x2), sc(y2))], fill=f, width=lw)
            else:
                _pil_dashed_line(draw, sc(x1), sc(y1), sc(x2), sc(y2),
                                 f, lw, (dash[0] * S, dash[1] * S))

        # --- Concentric filled rings ---
        for rr, col in [
            (R * 0.82, "#3B2B21"), (R * 0.62, "#473327"),
            (R * 0.44, "#534032"), (R * 0.28, "#604B3C"),
        ]:
            _e(cx - rr, cy - rr, cx + rr, cy + rr, fill=col)

        # --- Border ---
        draw.rectangle([S, S, sw - S, sh - S], outline=_pil_rgb(BORDER), width=S)

        # --- Distance rings ---
        for frac, col in [(0.33, "#5A4A36"), (0.66, "#6A5A45"), (1.00, "#7B6A53")]:
            rr = R * frac
            _e(cx - rr, cy - rr, cx + rr, cy + rr, outline=col)

        # --- Crosshair ---
        _dl(cx - R - 10, cy, cx + R + 10, cy, fill="#74624D", dash=(4, 4))
        _dl(cx, cy - R - 10, cx, cy + R + 10, fill="#74624D", dash=(4, 4))

        # --- Listener marker ---
        _e(cx - 15, cy - 15, cx + 15, cy + 15, fill="#2C2119", outline="#7A5C43")
        _e(cx - 8,  cy - 8,  cx + 8,  cy + 8,  fill="#3A2A1F", outline=ACCENT, width=2)

        # --- Sources ---
        nr = self.NODE_R
        for src in self.state.sources:
            if src.mute:
                continue

            live_pos = None
            if self._live_player is not None and src.recorded_movement:
                live_pos = self._live_player.display_positions.get(src.name)

            display_az = live_pos[0] if live_pos else src.azimuth
            display_el = live_pos[1] if live_pos else src.elevation
            x, y = self._source_to_xy_deg(display_az, display_el)
            col = src.color

            # Trajectory trail during live playback
            if live_pos is not None:
                trail_pts = [
                    self._source_to_xy_deg(azi_deg, ele_deg)
                    for _, azi_deg, ele_deg in src.recorded_movement
                ]
                for (tx1, ty1), (tx2, ty2) in zip(trail_pts, trail_pts[1:]):
                    _pil_dashed_line(draw, sc(tx1), sc(ty1), sc(tx2), sc(ty2),
                                     _pil_rgb(col), S, (3 * S, 5 * S))

            # Stem line from listener to node
            _pil_dashed_line(draw, sc(cx), sc(cy), sc(x), sc(y),
                             _pil_rgb(col), S, (3 * S, 3 * S))

            # Shadow halo
            _e(x - (nr + 7), y - (nr + 7), x + (nr + 7), y + (nr + 7), fill="#2B2119")

            # Elevation ring
            el_r = nr + 5 + (display_el / 90) * 11
            _pil_dashed_ellipse(draw,
                                sc(x - el_r), sc(y - el_r), sc(x + el_r), sc(y + el_r),
                                _pil_rgb(col), S, (2 * S, 3 * S))

            # Node circle with outline
            _e(x - nr, y - nr, x + nr, y + nr, fill=col, outline=TEXT, width=1.5)

            # Shine highlight
            draw.ellipse(
                [sc(x - nr + 3), sc(y - nr + 3), sc(x - nr + 6), sc(y - nr + 6)],
                fill=(255, 250, 242),
            )

        # --- Recording overlay (PIL part: trail + ring) ---
        if self._recording and self._record_source is not None:
            src = self._record_source
            if len(self._record_samples) >= 2:
                pts = [self._source_to_xy_deg(az, el) for _, az, el in self._record_samples]
                for (rx1, ry1), (rx2, ry2) in zip(pts, pts[1:]):
                    draw.line([(sc(rx1), sc(ry1)), (sc(rx2), sc(ry2))],
                              fill=(255, 107, 107), width=max(2, round(2 * S)))
            rx, ry = self._source_to_xy(src)
            draw.ellipse(
                [sc(rx - nr - 3), sc(ry - nr - 3), sc(rx + nr + 3), sc(ry + nr + 3)],
                outline=(255, 107, 107), width=max(2, round(2 * S)),
            )

        # --- Downscale with LANCZOS → smooth anti-aliased result ---
        img = img.resize((w, h), Image.LANCZOS)
        self._scene_photo = ImageTk.PhotoImage(img)
        self.create_image(0, 0, image=self._scene_photo, anchor="nw")

        # --- Text labels (native Tkinter — crisp on all platforms) ---
        label_col = TEXT_DIM
        self.create_text(cx, cy - R - 40, text="FRONT", fill=label_col, font=("Helvetica", 9, "bold"), anchor="s")
        self.create_text(cx, cy + R + 40, text="BACK",  fill=label_col, font=("Helvetica", 9, "bold"), anchor="n")
        self.create_text(cx - R - 40, cy, text="LEFT",  fill=label_col, font=("Helvetica", 9, "bold"), anchor="e")
        self.create_text(cx + R + 40, cy, text="RIGHT", fill=label_col, font=("Helvetica", 9, "bold"), anchor="w")
        self.create_text(cx, cy, text="•", fill=ACCENT, font=("Helvetica", 14, "bold"))

        for src in self.state.sources:
            if src.mute:
                continue
            live_pos = None
            if self._live_player is not None and src.recorded_movement:
                live_pos = self._live_player.display_positions.get(src.name)
            display_az = live_pos[0] if live_pos else src.azimuth
            display_el = live_pos[1] if live_pos else src.elevation
            x, y = self._source_to_xy_deg(display_az, display_el)
            self.create_text(x, y - nr - 10, text=src.name,
                             fill=TEXT, font=("Helvetica", 8, "bold"), anchor="s")
            self.create_text(x, y + nr + 10, text=f"{display_az:+.0f}°",
                             fill=TEXT_DIM, font=("Courier", 8), anchor="n")
            if src.recorded_movement and src is not self._record_source:
                self.create_text(x + nr + 6, y - nr - 6, text="↻",
                                 fill=ACCENT, font=("Helvetica", 11, "bold"), anchor="center")

        if self._recording:
            self.create_text(cx, 18,
                             text="● RECORDING — click the button again to stop",
                             fill="#FF6B6B", font=("Helvetica", 10, "bold"), anchor="n")

    def _on_press(self, event):
        if self._recording:
            self._record_move_to(event.x, event.y)
            return

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
        if self._recording:
            self._record_move_to(event.x, event.y)
            return

        if self._drag is None:
            return

        src = self.state.sources[self._drag]
        if src.recorded_movement is not None:
            src.recorded_movement = None
        src.azimuth = self._xy_to_az(event.x, event.y)
        self.redraw()

        if self._rows:
            self._rows[self._drag].refresh_az()

        if self._inspector is not None and self._inspector.source is src:
            self._inspector.update_azimuth()

    def _on_release(self, _event):
        self._drag = None

    def _record_move_to(self, x, y):
        """
        Called on every click/drag event while recording. Moves the source
        being recorded to the clicked/dragged point — this is what allows
        both continuous gestures (circular motion, drag left-to-right) and
        instant "teleports" (single clicks at different spots, so the source
        appears to jump/disappear-and-reappear elsewhere).
        """
        src = self._record_source
        if src is None:
            return

        azimuth = self._xy_to_az(x, y)

        # Elevation follows distance from centre (same mapping used for display):
        # near the centre = high elevation (overhead), near the edge = horizon.
        cx, cy = self._center()
        r = self._effective_radius()
        dist = math.hypot(x - cx, y - cy)
        elevation = max(0.0, min(90.0, 90.0 * (1.0 - dist / r))) if r > 0 else 0.0

        src.azimuth = azimuth
        src.elevation = elevation

        t = time.perf_counter() - self._record_start_time
        self._record_samples.append((t, azimuth, elevation))

        self.redraw()

        if self._rows:
            for row in self._rows:
                if row.source is src:
                    row.refresh_az()

        if self._inspector is not None and self._inspector.source is src:
            self._inspector.update_azimuth()
            self._inspector.update_elevation_display()


class OutputPanel(tk.Frame):
    def __init__(
        self,
        parent,
        state: AppState,
        status_ref=None,
        scene_ref=None,
        inspector_ref=None,
        rows_ref=None,
        **kwargs,
    ):
        super().__init__(parent, bg=PANEL_BG, **kwargs)
        self.state = state
        self._status_ref = status_ref
        self._scene_ref = scene_ref
        self._inspector_ref = inspector_ref
        self._rows_ref = rows_ref or []
        self._build()

    def _build(self):
        s = self.state

        def _add_icon(btn, icon_name, icon_size=13, icon_fg=TEXT):
            try:
                col = _pil_rgb(icon_fg)
                icon = _load_svg_icon(str(ICONS_DIR / f"{icon_name}.svg"), icon_size, col)
                photo = ImageTk.PhotoImage(icon)
                btn.config(image=photo, compound="left", padx=6)
                btn._icon_ref = photo
            except Exception:
                pass

        def card(title):
            tk.Label(
                self, text=title,
                bg=PANEL_BG, fg=TEXT_DIM,
                font=("Helvetica", 8),
                anchor="w",
            ).pack(fill="x", padx=16, pady=(14, 4))
            frame = tk.Frame(self, bg=PANEL_BG2, highlightthickness=1, highlightbackground=BORDER)
            frame.pack(fill="x", padx=10, pady=(0, 8))
            inner = tk.Frame(frame, bg=PANEL_BG2)
            inner.pack(fill="x", padx=12, pady=10)
            return inner

        def field_label(parent, text):
            tk.Label(
                parent, text=text,
                bg=PANEL_BG2, fg=TEXT_DIM,
                font=("Helvetica", 8),
                anchor="w",
            ).pack(anchor="w", pady=(0, 2))

        def divider(parent):
            tk.Frame(parent, bg="#3A2B1E", height=1).pack(fill="x", pady=8)

        # INPUT
        inp = card("INPUT")

        # Song file row: truncated name + Browse button on the right
        song_row = tk.Frame(inp, bg=PANEL_BG2)
        song_row.pack(fill="x", pady=(0, 8))
        self._song_lbl = tk.Label(
            song_row,
            text="no song selected",
            bg=PANEL_BG2, fg=TEXT_DIM,
            font=FONT_SMALL,
            anchor="w", justify="left", wraplength=210,
        )
        self._song_lbl.pack(side="left", fill="x", expand=True)
        browse_btn = tk.Button(
            song_row, text="Browse…",
            font=FONT_SMALL, command=self._pick_song,
        )
        make_button_3d(browse_btn, PANEL_BG, active_bg="#2E2820", pressed_bg="#1C1916")
        _add_icon(browse_btn, "browse")
        browse_btn.pack(side="right", padx=(6, 0))

        # Two equal-width action buttons: Demix | Load Stems
        btn_grid = tk.Frame(inp, bg=PANEL_BG2)
        btn_grid.pack(fill="x")
        btn_grid.grid_columnconfigure(0, weight=1, uniform="action_btns")
        btn_grid.grid_columnconfigure(1, weight=1, uniform="action_btns")

        self._demix_btn = tk.Button(
            btn_grid, text="Demix",
            font=("Helvetica", 10, "bold"),
            command=self._on_demix,
        )
        make_button_3d(self._demix_btn, ACCENT2, fg="#241B15",
                       border=BORDER, active_bg="#96B87A", pressed_bg="#6E8E59")
        _add_icon(self._demix_btn, "demix", icon_fg="#241B15")
        self._demix_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        load_stems_btn = tk.Button(
            btn_grid, text="Load Stems",
            font=("Helvetica", 10, "bold"),
            command=self._pick_stems_folder,
        )
        make_button_3d(load_stems_btn, ACCENT2, fg="#241B15",
                       border=BORDER, active_bg="#96B87A", pressed_bg="#6E8E59")
        _add_icon(load_stems_btn, "load-stems", icon_fg="#241B15")
        load_stems_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        # RENDERING
        rnd = card("RENDERING")

        # Mode dropdown
        field_label(rnd, "Mode")
        renderer_choices = {"Binaural": "binaural", "Layout Speaker": "layout_speaker"}
        current_label = next(
            (lbl for lbl, val in renderer_choices.items() if val == s.renderer),
            "Binaural",
        )
        self._renderer_menu_var = tk.StringVar(value=current_label)
        self._renderer_var = tk.StringVar(value=renderer_choices[self._renderer_menu_var.get()])
        renderer_menu = tk.OptionMenu(
            rnd, self._renderer_menu_var, *renderer_choices.keys(),
            command=lambda sel: self._set_renderer_from_menu(renderer_choices[sel]),
        )
        renderer_menu.config(
            bg=PANEL_BG, fg=TEXT, activebackground="#2E2820", activeforeground=TEXT,
            relief="flat", bd=0, highlightthickness=1, highlightbackground=BORDER,
            font=FONT_SMALL, anchor="w", padx=8, pady=4,
        )
        renderer_menu["menu"].config(
            bg=PANEL_BG2, fg=TEXT, activebackground=ACCENT,
            activeforeground="white", font=FONT_SMALL, bd=0,
        )
        renderer_menu.pack(fill="x", pady=(0, 0))

        divider(rnd)

        # Speaker layout: label + inline Load button
        field_label(rnd, "Speaker layout")
        layout_row = tk.Frame(rnd, bg=PANEL_BG2)
        layout_row.pack(fill="x", pady=(0, 6))
        self._layout_lbl = tk.Label(
            layout_row, text="default (museum 17ch)",
            bg=PANEL_BG2, fg=TEXT,
            font=FONT_SMALL, anchor="w", justify="left", wraplength=200,
        )
        self._layout_lbl.pack(side="left", fill="x", expand=True)
        load_csv_btn = tk.Button(
            layout_row, text="Load…",
            font=FONT_SMALL, command=self._pick_layout,
        )
        make_button_3d(load_csv_btn, PANEL_BG, active_bg="#2E2820", pressed_bg="#1C1916")
        _add_icon(load_csv_btn, "load")
        load_csv_btn.pack(side="right", padx=(6, 0))

        # HRTF: label + inline Load button
        field_label(rnd, "HRTF")
        hrtf_row = tk.Frame(rnd, bg=PANEL_BG2)
        hrtf_row.pack(fill="x", pady=(0, 6))
        self._hrtf_lbl = tk.Label(
            hrtf_row, text="default HRTF",
            bg=PANEL_BG2, fg=TEXT,
            font=FONT_SMALL, anchor="w", justify="left", wraplength=200,
        )
        self._hrtf_lbl.pack(side="left", fill="x", expand=True)
        load_sofa_btn = tk.Button(
            hrtf_row, text="Load…",
            font=FONT_SMALL, command=self._pick_hrtf,
        )
        make_button_3d(load_sofa_btn, PANEL_BG, active_bg="#2E2820", pressed_bg="#1C1916")
        _add_icon(load_sofa_btn, "load")
        load_sofa_btn.pack(side="right", padx=(6, 0))

        divider(rnd)

        # HOA Order — spinbox (1–7; orders 1–3 most common)
        order_row = tk.Frame(rnd, bg=PANEL_BG2)
        order_row.pack(fill="x")
        tk.Label(
            order_row, text="HOA order",
            bg=PANEL_BG2, fg=TEXT_DIM,
            font=("Helvetica", 8), anchor="w",
        ).pack(side="left")
        tk.Label(
            order_row, text="(1–3 most common)",
            bg=PANEL_BG2, fg="#6A5A4A",
            font=("Helvetica", 7), anchor="w",
        ).pack(side="left", padx=(6, 0))
        self._order_var = tk.StringVar(value=str(s.hoa_order))
        order_spin = tk.Spinbox(
            order_row,
            from_=1, to=7, increment=1,
            textvariable=self._order_var,
            width=3,
            bg=PANEL_BG, fg=TEXT,
            insertbackground=TEXT,
            buttonbackground=PANEL_BG2,
            relief="flat", bd=0,
            highlightthickness=1, highlightbackground=BORDER,
            highlightcolor=ACCENT,
            font=FONT_SMALL,
            justify="center",
            command=self._on_order_change,
        )
        order_spin.pack(side="right")
        order_spin.bind("<KeyRelease>", lambda _e: self._on_order_change())

        # OUTPUT
        out = card("OUTPUT")

        # Output directory
        field_label(out, "Directory")
        dir_row = tk.Frame(out, bg=PANEL_BG2)
        dir_row.pack(fill="x", pady=(0, 8))
        self._outdir_lbl = tk.Label(
            dir_row, text=str(s.out_dir),
            bg=PANEL_BG2, fg=TEXT,
            font=FONT_SMALL, anchor="w", justify="left", wraplength=220,
        )
        self._outdir_lbl.pack(side="left", fill="x", expand=True)
        change_btn = tk.Button(
            dir_row, text="Change…",
            font=FONT_SMALL, command=self._pick_outdir,
        )
        make_button_3d(change_btn, PANEL_BG, active_bg="#2E2820", pressed_bg="#1C1916")
        _add_icon(change_btn, "browse")
        change_btn.pack(side="right", padx=(6, 0))

        # Filename entry
        field_label(out, "Filename")
        self._output_name_var = tk.StringVar(value=getattr(s, "output_name", ""))
        name_entry = tk.Entry(
            out, textvariable=self._output_name_var,
            bg=PANEL_BG, fg=TEXT, insertbackground=TEXT,
            relief="flat", bd=0, font=FONT_SMALL,
            highlightthickness=1, highlightbackground=BORDER,
            highlightcolor=ACCENT,
        )
        name_entry.pack(fill="x", ipady=5)
        name_entry.bind("<KeyRelease>", self._on_output_name_change)

    def _pick_song(self):
        p = filedialog.askopenfilename(
            title="Select song file",
            filetypes=[("Audio files", "*.wav *.mp3 *.flac"), ("All files", "*.*")],
        )
        if p:
            self.state.song_path = p
            self._song_lbl.config(text=Path(p).name)

    def _pick_stems_folder(self):
        folder = filedialog.askdirectory(title="Select stems folder")
        if not folder:
            return

        folder_path = Path(folder)
        folder_name = folder_path.name
        song_name = folder_name.removesuffix("-stems")
        self.state.song_path = str(folder_path / f"{song_name}.wav")
        self._song_lbl.config(text=f"{folder_name} (stems)")

        wav_files = list(folder_path.glob("*.wav"))
        if not wav_files:
            messagebox.showwarning("No WAV files", "The selected folder contains no .wav files.")
            return

        valid_source_names = {src.name for src in self.state.sources}
        stems = {}
        for wav in wav_files:
            name = wav.stem.lower()
            for source_name in valid_source_names:
                if (
                    name == source_name
                    or name.endswith(f"_{source_name}")
                    or name.endswith(f"-{source_name}")
                    or name.startswith(f"{source_name}_")
                    or name.startswith(f"{source_name}-")
                ):
                    stems[source_name] = str(wav)
                    break

        if not stems:
            messagebox.showwarning(
                "No stems recognized",
                "No recognized stems found.\nExpected names like vocals, drums, bass, guitar, piano, other.",
            )
            return

        populate_sources_from_stem_paths(self.state, stems)

        if self._inspector_ref is not None and self._inspector_ref.source is not None:
            self._inspector_ref.set_source(self._inspector_ref.source)

        for row in self._rows_ref:
            row.refresh_all()

        if self._scene_ref is not None:
            self._scene_ref.redraw()

        loaded_names = ", ".join(sorted(stems.keys()))
        messagebox.showinfo("Stems loaded", f"Loaded stems: {loaded_names}")

    def _on_demix(self):
        t = threading.Thread(
            target=run_demix_and_populate,
            args=(self.state, self._status_ref, self._demix_btn, self._after_demix),
            daemon=True,
        )
        t.start()

    def _after_demix(self):
        for row in self._rows_ref:
            row.refresh_all()

        if self._inspector_ref is not None and self._inspector_ref.source is not None:
            self._inspector_ref.set_source(self._inspector_ref.source)

        if self._scene_ref is not None:
            self._scene_ref.redraw()

    def _on_order_change(self):
        try:
            v = int(self._order_var.get())
            if 1 <= v <= 7:
                self.state.hoa_order = v
        except ValueError:
            pass

    def _on_renderer(self):
        self.state.renderer = self._renderer_var.get()

    def _set_renderer_from_menu(self, value: str):
        self.state.renderer = value

    def _on_output_name_change(self, _event=None):
        self.state.output_name = self._output_name_var.get()

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
        super().__init__(parent, bg=BG_2, height=26, **kwargs)
        self._var = tk.StringVar(value="Ready.")
        tk.Label(
            self,
            textvariable=self._var,
            bg=BG_2,
            fg=TEXT_DIM,
            font=FONT_SMALL,
            anchor="w",
        ).pack(side="left", padx=10)

    def set(self, msg: str):
        self._var.set(msg)
        self.update_idletasks()