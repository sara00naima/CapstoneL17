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

    def _build(self):
        s = self.state

        topbar = tk.Frame(self, bg=BG, height=70)
        topbar.pack(fill="x", side="top")
        topbar.pack_propagate(False)

        title_wrap = tk.Frame(topbar, bg=BG)
        title_wrap.pack(side="left", padx=14, pady=10)

        tk.Label(
            title_wrap,
            text="SPATIAL AUDIO PIPELINE",
            bg=BG,
            fg=TEXT,
            font=("Helvetica", 18, "bold"),
            anchor="w",
        ).pack(anchor="w")

        tk.Label(
            title_wrap,
            text="Drag stems in space · inspect one source at a time · choose renderer and output",
            bg=BG,
            fg=TEXT_DIM,
            font=("Helvetica", 10),
            anchor="w",
        ).pack(anchor="w", pady=(3, 0))

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
        self._gen_btn.pack(side="right", padx=14, pady=10)

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

        centre = tk.Frame(body, bg=BG)
        centre.grid(row=0, column=1, sticky="nsew", padx=2, pady=8)

        right = tk.Frame(
            body,
            bg=PANEL_BG,
            bd=0,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        right.grid(row=0, column=2, sticky="nsew", padx=(4, 8), pady=8)

        tk.Label(
            left,
            text="SOURCES",
            bg=PANEL_BG,
            fg=ACCENT,
            font=FONT_SECTION,
        ).pack(anchor="w", padx=14, pady=(14, 8))

        rows_wrap = tk.Frame(left, bg=PANEL_BG)
        rows_wrap.pack(fill="x", padx=10, pady=(0, 8))

        tk.Label(
            centre,
            text="SCENE VIEW",
            bg=BG,
            fg=ACCENT,
            font=FONT_SECTION,
        ).pack(anchor="w", padx=10, pady=(2, 4))

        tk.Label(
            centre,
            text="Drag a node to change azimuth. Elevation is edited in the source inspector.",
            bg=BG,
            fg=TEXT_DIM,
            font=FONT_SMALL,
        ).pack(anchor="w", padx=10, pady=(0, 8))

        scene_view = SceneView(centre, s)
        scene_view.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        inspector = SourceInspector(left, scene_view)
        inspector.pack(fill="x", padx=10, pady=(6, 12))

        rows = []
        for src in s.sources:
            row = SourceRow(rows_wrap, src, scene_view, on_select=inspector.set_source)
            row.pack(fill="x", padx=4, pady=4)
            rows.append(row)

        if s.sources:
            inspector.set_source(s.sources[0])

        scene_view.set_rows(rows)
        scene_view.set_inspector(inspector)
        scene_view.after(100, scene_view.redraw)

        tk.Label(
            right,
            text="OUTPUT",
            bg=PANEL_BG,
            fg=ACCENT,
            font=FONT_SECTION,
        ).pack(anchor="w", padx=14, pady=(14, 8))

        OutputPanel(right, s,
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