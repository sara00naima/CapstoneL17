import threading
import tkinter as tk

from gui_backend import (
    AppState,
    BG,
    PANEL_BG,
    ACCENT,
    BORDER,
    TEXT,
    TEXT_DIM,
    FONT_APP_TITLE,
    FONT_SECTION,
    FONT_SMALL,
    run_generate,
)
from gui_widgets import SourceRow, SceneView, OutputPanel, StatusBar


class SpatialAudioGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Spatial Audio Pipeline")
        self.configure(bg=BG)
        self.geometry("1220x760")
        self.minsize(980, 640)

        self.state = AppState()
        self._build()

    def _build(self):
        s = self.state

        topbar = tk.Frame(self, bg=BG, height=52)
        topbar.pack(fill="x", side="top")
        topbar.pack_propagate(False)

        title_wrap = tk.Frame(topbar, bg=BG)
        title_wrap.pack(side="left", padx=16, pady=8)

        tk.Label(
            title_wrap,
            text="SPATIAL AUDIO PIPELINE",
            bg=BG,
            fg=TEXT,
            font=FONT_APP_TITLE,
            anchor="w",
        ).pack(anchor="w")

        tk.Label(
            title_wrap,
            text="Drag stems in space · set elevation · choose renderer/output",
            bg=BG,
            fg=TEXT_DIM,
            font=FONT_SMALL,
            anchor="w",
        ).pack(anchor="w")

        self._gen_btn = tk.Button(
            topbar,
            text="▶  GENERATE",
            bg=ACCENT,
            fg="white",
            activebackground="#ff7690",
            activeforeground="white",
            relief="flat",
            bd=0,
            font=("Helvetica", 10, "bold"),
            padx=20,
            pady=8,
            cursor="hand2",
            command=self._on_generate,
        )
        self._gen_btn.pack(side="right", padx=16, pady=8)

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", side="top")

        self._status = StatusBar(self)
        self._status.pack(fill="x", side="bottom")

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        left = tk.Frame(
            body,
            bg=PANEL_BG,
            width=420,
            bd=0,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        left.pack(side="left", fill="y", padx=(10, 6), pady=10)
        left.pack_propagate(False)

        centre = tk.Frame(body, bg=BG)
        centre.pack(side="left", fill="both", expand=True, padx=4, pady=10)

        right = tk.Frame(
            body,
            bg=PANEL_BG,
            width=320,
            bd=0,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        right.pack(side="right", fill="y", padx=(6, 10), pady=10)
        right.pack_propagate(False)

        tk.Label(
            left,
            text="SOURCES",
            bg=PANEL_BG,
            fg=ACCENT,
            font=FONT_SECTION,
        ).pack(anchor="w", padx=12, pady=(12, 6))

        rows_wrap = tk.Frame(left, bg=PANEL_BG)
        rows_wrap.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        tk.Label(
            centre,
            text="SCENE VIEW",
            bg=BG,
            fg=ACCENT,
            font=FONT_SECTION,
        ).pack(anchor="w", padx=10, pady=(2, 4))

        tk.Label(
            centre,
            text="Drag the sources with the mouse. Elevation is controlled from the sliders.",
            bg=BG,
            fg=TEXT_DIM,
            font=FONT_SMALL,
        ).pack(anchor="w", padx=10, pady=(0, 8))

        scene_view = SceneView(centre, s)
        scene_view.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        rows = []
        for src in s.sources:
            row = SourceRow(rows_wrap, src, scene_view)
            row.pack(fill="x", padx=4, pady=4)
            rows.append(row)

        scene_view.set_rows(rows)
        scene_view.after(100, scene_view.redraw)

        tk.Label(
            right,
            text="OUTPUT",
            bg=PANEL_BG,
            fg=ACCENT,
            font=FONT_SECTION,
        ).pack(anchor="w", padx=12, pady=(12, 4))

        OutputPanel(right, s).pack(fill="both", expand=True)

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