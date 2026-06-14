"""
GUI application for the spatial audio pipeline.
Usage (from project root):
python src/gui/gui_app.py
"""

import threading
import tkinter as tk

from gui_backend import (
    AppState,
    BG,
    PANEL_BG,
    PANEL_BG2,
    ACCENT,
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
)


class SpatialAudioGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Spatial Audio Pipeline")
        self.configure(bg=BG)
        self.geometry("1280x800")
        self.minsize(1080, 700)

        self.state = AppState()
        self._build()

    def _panel_header(self, parent, text):
        wrap = tk.Frame(
            parent,
            bg=PANEL_BG2,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        wrap.pack(fill="x", padx=8, pady=(8, 6))

        tk.Frame(wrap, bg=ACCENT, width=6).pack(side="left", fill="y")

        tk.Label(
            wrap,
            text=text,
            bg=PANEL_BG2,
            fg=TEXT,
            font=("Helvetica", 12, "bold"),
            anchor="w",
            padx=12,
            pady=8,
        ).pack(side="left", fill="x", expand=True)

        return wrap

    def _build(self):
        s = self.state

        topbar = tk.Frame(self, bg=BG, height=70)
        topbar.pack(fill="x", side="top")
        topbar.pack_propagate(False)

        left_spacer = tk.Frame(topbar, bg=BG, width=170)
        left_spacer.pack(side="left")

        self._gen_btn = tk.Button(
            topbar,
            text="▸ GENERATE",
            bg=ACCENT,
            fg="white",
            activebackground="#ff7690",
            activeforeground="white",
            relief="flat",
            bd=0,
            font=("Helvetica", 11, "bold"),
            padx=20,
            pady=10,
            cursor="hand2",
            command=self._on_generate,
        )
        self._gen_btn.pack(side="right", padx=14, pady=8)

        title_wrap = tk.Frame(topbar, bg=BG)
        title_wrap.pack(side="left", fill="both", expand=True)

        tk.Label(
            title_wrap,
            text="SPATIAL AUDIO PIPELINE",
            bg=BG,
            fg=TEXT,
            font=FONT_APP_TITLE,
            anchor="center",
            justify="center",
        ).pack(expand=True)

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", side="top")

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

    def _on_generate(self):
        t = threading.Thread(
            target=run_generate,
            args=(self.state, self._status, self._gen_btn),
            daemon=True,
        )
        t.start()


if __name__ == "__main__":
    app = SpatialAudioGUI()
    app.mainloop()