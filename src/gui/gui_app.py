"""
GUI application for the spatial audio pipeline.
Usage (from project root):
python src/gui/gui_app.py
"""

import threading
import tkinter as tk

try:
    import sounddevice as _sd  # noqa: F401 — import check only
    _PLAYBACK_AVAILABLE = True
except ImportError:
    _PLAYBACK_AVAILABLE = False

from gui_backend import (
    AppState,
    BG,
    PANEL_BG,
    PANEL_BG2,
    ACCENT,
    ACCENT2,
    BORDER,
    TEXT,
    TEXT_DIM,
    FONT_APP_TITLE,
    FONT_SMALL,
    run_generate,
)
from gui_widgets import (
    SourceRow,
    SourceInspector,
    SceneView,
    OutputPanel,
    StatusBar,
    make_button_3d,
)


class SpatialAudioGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("3D Audio Generator")
        self.configure(bg=BG)
        self.geometry("1280x800")
        self.minsize(1080, 700)

        self.state = AppState()
        self._generate_item = None
        self._title_item = None
        self._topbar_gradient = None
        self._play_item = None
        self._live_player = None
        self._build()

    def _panel_header(self, parent, text):
        wrap = tk.Frame(
            parent,
            bg=PANEL_BG2,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        wrap.pack(fill="x", padx=8, pady=(8, 6))

        tk.Label(
            wrap,
            text=text,
            bg=PANEL_BG2,
            fg=TEXT,
            font=("Helvetica", 12, "bold"),
            anchor="center",
            padx=12,
            pady=8,
        ).pack(side="left", fill="x", expand=True)

        return wrap

    def _hex_to_rgb(self, value):
        value = value.lstrip("#")
        return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))

    def _make_vertical_gradient(self, width, height, color_top, color_bottom):
        img = tk.PhotoImage(width=width, height=height)

        r1, g1, b1 = self._hex_to_rgb(color_top)
        r2, g2, b2 = self._hex_to_rgb(color_bottom)

        for y in range(height):
            t = y / max(1, height - 1)
            r = int(r1 + (r2 - r1) * t)
            g = int(g1 + (g2 - g1) * t)
            b = int(b1 + (b2 - b1) * t)
            color = f"#{r:02x}{g:02x}{b:02x}"
            img.put(color, to=(0, y, width, y + 1))

        return img

    def _redraw_topbar(self, event=None):
        canvas = self._topbar
        w = max(1, canvas.winfo_width())
        h = max(1, canvas.winfo_height())

        self._topbar_gradient = self._make_vertical_gradient(
            w,
            h,
            "#b88d5c",
            "#d8b58b",
        )

        canvas.delete("gradient")
        canvas.create_image(0, 0, image=self._topbar_gradient, anchor="nw", tags="gradient")
        canvas.tag_lower("gradient")

        canvas.coords(self._title_item, w // 2, h // 2)
        canvas.coords(self._generate_item, w - 95, h // 2)
        canvas.coords(self._play_item, 95, h // 2)

    def _on_generate_enter(self, _event=None):
        self._topbar.itemconfig(self._generate_item, image=self._gen_img_hover)

    def _on_generate_leave(self, _event=None):
        self._topbar.itemconfig(self._generate_item, image=self._gen_img_normal)

    def _on_generate_press(self, _event=None):
        self._topbar.itemconfig(self._generate_item, image=self._gen_img_pressed)

    def _on_generate_release(self, event):
        x = self._topbar.canvasx(event.x)
        y = self._topbar.canvasy(event.y)
        bbox = self._topbar.bbox(self._generate_item)

        if bbox is None:
            return

        x1, y1, x2, y2 = bbox
        if x1 <= x <= x2 and y1 <= y <= y2:
            self._topbar.itemconfig(self._generate_item, image=self._gen_img_hover)
            self._on_generate()
        else:
            self._topbar.itemconfig(self._generate_item, image=self._gen_img_normal)

    def _build(self):
        s = self.state

        self._topbar = tk.Canvas(
            self,
            bg=BG,
            height=100,
            highlightthickness=0,
            bd=0,
        )
        self._topbar.pack(fill="x", side="top")
        self._topbar.bind("<Configure>", self._redraw_topbar)

        self._gen_img_normal = tk.PhotoImage(file="assets/generate_normal.png").subsample(2, 2)
        self._gen_img_hover = tk.PhotoImage(file="assets/generate_hover.png").subsample(2, 2)
        self._gen_img_pressed = tk.PhotoImage(file="assets/generate_pressed.png").subsample(2, 2)
        self._title_img = tk.PhotoImage(file="assets/title_logo.png").zoom(2, 2).subsample(3, 3)

        self._title_item = self._topbar.create_image(0, 0, image=self._title_img, anchor="center")
        self._generate_item = self._topbar.create_image(0, 0, image=self._gen_img_normal, anchor="center")

        self._topbar.tag_bind(self._generate_item, "<Enter>", self._on_generate_enter)
        self._topbar.tag_bind(self._generate_item, "<Leave>", self._on_generate_leave)
        self._topbar.tag_bind(self._generate_item, "<ButtonPress-1>", self._on_generate_press)
        self._topbar.tag_bind(self._generate_item, "<ButtonRelease-1>", self._on_generate_release)

        self._play_btn = tk.Button(
            self._topbar,
            text="▶  Play",
            font=("Helvetica", 10, "bold"),
            bg=ACCENT2,
            fg="#241B15",
            activebackground="#96B87A",
            activeforeground="#241B15",
            relief="raised",
            bd=1,
            highlightthickness=1,
            highlightbackground=BORDER,
            padx=12,
            pady=6,
            cursor="hand2",
            command=self._on_play_stop,
        )
        self._play_item = self._topbar.create_window(0, 0, window=self._play_btn, anchor="center")

        self._status = StatusBar(self)
        self._status.pack(fill="x", side="bottom")

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=0, minsize=360)
        body.grid_columnconfigure(1, weight=1, minsize=480)
        body.grid_columnconfigure(2, weight=0, minsize=300)

        left = tk.Frame(
            body,
            bg=PANEL_BG,
            bd=0,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        left.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)

        centre = tk.Frame(
            body,
            bg=PANEL_BG,
            bd=0,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        centre.grid(row=0, column=1, sticky="nsew", padx=2, pady=8)

        right = tk.Frame(
            body,
            bg=PANEL_BG,
            bd=0,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        right.grid(row=0, column=2, sticky="nsew", padx=(4, 8), pady=8)

        self._panel_header(left, "SOURCES")
        rows_wrap = tk.Frame(left, bg=PANEL_BG)
        rows_wrap.pack(fill="x", padx=10, pady=(0, 8))

        self._panel_header(centre, "SCENE VIEW")
        tk.Label(
            centre,
            text="Drag a node to change azimuth. Elevation is edited in the source inspector.",
            bg=PANEL_BG,
            fg=TEXT_DIM,
            font=FONT_SMALL,
        ).pack(anchor="w", padx=12, pady=(0, 8))

        scene_view = SceneView(centre, s)
        scene_view.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        record_bar = tk.Frame(centre, bg=PANEL_BG)
        record_bar.pack(fill="x", padx=8, pady=(0, 8))

        record_btn = tk.Button(
            record_bar,
            text="● Record Movement",
            font=("Helvetica", 10, "bold"),
            command=lambda: scene_view.toggle_recording(),
        )
        make_button_3d(
            record_btn,
            ACCENT2,
            fg="#241B15",
            border=BORDER,
            active_bg="#96B87A",
            pressed_bg="#6E8E59",
        )
        record_btn.pack(side="left")

        clear_btn = tk.Button(
            record_bar,
            text="Clear Movement",
            font=FONT_SMALL,
            command=lambda: self._on_clear_movement(scene_view),
        )
        make_button_3d(clear_btn, PANEL_BG2, active_bg=ACCENT, pressed_bg="#123457")
        clear_btn.pack(side="left", padx=(8, 0))

        tk.Label(
            record_bar,
            text="Click to start, move/click the node, click again to stop.\n"
                 "The gesture loops for the whole rendered file.",
            bg=PANEL_BG,
            fg=TEXT_DIM,
            font=FONT_SMALL,
            justify="left",
        ).pack(side="left", padx=(12, 0))

        scene_view.set_record_button(record_btn)

        inspector = SourceInspector(left, s, scene_view)
        inspector.pack(fill="x", padx=10, pady=(6, 8))

        rows = []
        inspector.rows = rows
        for src in s.sources:
            row = SourceRow(rows_wrap, src, scene_view, on_select=inspector.set_source)
            row.pack(fill="x", padx=4, pady=4)
            rows.append(row)

        if s.sources:
            inspector.set_source(s.sources[0])

        scene_view.set_rows(rows)
        scene_view.set_inspector(inspector)
        scene_view.after(100, scene_view.redraw)

        self._panel_header(right, "OUTPUT")
        OutputPanel(
            right,
            s,
            status_ref=self._status,
            scene_ref=scene_view,
            inspector_ref=inspector,
            rows_ref=rows,
        ).pack(fill="both", expand=True, padx=2, pady=(0, 8))

    def _on_clear_movement(self, scene_view):
        inspector_source = scene_view.get_selected_source()

        if inspector_source is None:
            return

        scene_view.clear_recorded_movement(inspector_source)

    def _on_play_stop(self):
        from tkinter import messagebox

        if not _PLAYBACK_AVAILABLE:
            messagebox.showerror(
                "Missing dependency",
                "Install sounddevice to enable playback:\n  pip install sounddevice",
            )
            return

        if self._live_player is not None:
            self._live_player.stop()
            self._live_player = None
            self._play_btn.config(text="▶  Play")
            return

        # Load HRTF + stems in a background thread to avoid freezing the UI
        self._play_btn.config(text="Loading…", state="disabled")

        def _start():
            try:
                from spatial_pipeline.live_player import LivePlayer
                from spatial_pipeline.config import DEFAULT_HRTF_SOFA

                sofa = str(self.state.hrtf_path) if self.state.hrtf_path else str(DEFAULT_HRTF_SOFA)
                player = LivePlayer(self.state, sofa)
                player.start()
                self._live_player = player
                self._play_btn.after(0, lambda: self._play_btn.config(
                    text="■  Stop", state="normal"
                ))
            except Exception as e:
                self._live_player = None
                self._play_btn.after(0, lambda: self._play_btn.config(
                    text="▶  Play", state="normal"
                ))
                self._play_btn.after(0, lambda: messagebox.showerror("Playback error", str(e)))

        threading.Thread(target=_start, daemon=True).start()

    def _on_generate(self):
        fake_btn = _CanvasButtonProxy(
            on_config=lambda **kwargs: self.after(0, self._apply_generate_state, kwargs)
        )

        t = threading.Thread(
            target=run_generate,
            args=(self.state, self._status, fake_btn),
            daemon=True,
        )
        t.start()

    def _apply_generate_state(self, kwargs):
        state = kwargs.get("state")
        if state == "disabled":
            self._topbar.itemconfig(self._generate_item, image=self._gen_img_pressed)
        elif state == "normal":
            self._topbar.itemconfig(self._generate_item, image=self._gen_img_normal)


class _CanvasButtonProxy:
    def __init__(self, on_config):
        self._on_config = on_config

    def config(self, **kwargs):
        self._on_config(**kwargs)

    configure = config


if __name__ == "__main__":
    app = SpatialAudioGUI()
    app.mainloop()