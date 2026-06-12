import threading
import tkinter as tk

from gui_backend import AppState, BG, PANEL_BG, ACCENT, ACCENT2, TEXT, TEXT_DIM, run_generate
from gui_widgets import SourceRow, SceneView, OutputPanel, StatusBar


class SpatialAudioGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Spatial Audio Pipeline")
        self.configure(bg=BG)
        self.geometry("1180x720")
        self.minsize(900, 600)

        self.state = AppState()
        self._build()

    def _build(self):
        s = self.state

        topbar = tk.Frame(self, bg=BG, height=44)
        topbar.pack(fill="x", side="top")
        topbar.pack_propagate(False)

        tk.Label(
            topbar, text="SPATIAL AUDIO PIPELINE", bg=BG, fg=TEXT,
            font=("Helvetica", 13, "bold")
        ).pack(side="left", padx=16, pady=10)

        tk.Label(
            topbar, text="museum edition", bg=BG, fg=TEXT_DIM,
            font=("Helvetica", 9)
        ).pack(side="left")

        self._gen_btn = tk.Button(
            topbar, text="▶  GENERATE",
            bg=ACCENT, fg="white", relief="flat",
            font=("Helvetica", 10, "bold"),
            padx=18, pady=6, cursor="hand2",
            command=self._on_generate,
        )
        self._gen_btn.pack(side="right", padx=16, pady=7)

        self._status = StatusBar(self)
        self._status.pack(fill="x", side="bottom")

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        left = tk.Frame(body, bg=PANEL_BG, width=380)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        tk.Label(
            left, text="SOURCES", bg=PANEL_BG, fg=ACCENT,
            font=("Helvetica", 9, "bold")
        ).pack(anchor="w", padx=10, pady=(10, 4))

        centre = tk.Frame(body, bg=BG)
        centre.pack(side="left", fill="both", expand=True)

        tk.Label(
            centre, text="SCENE VIEW  (drag sources · elevation via slider)",
            bg=BG, fg=ACCENT, font=("Helvetica", 9, "bold")
        ).pack(anchor="w", padx=10, pady=(10, 2))

        scene_view = SceneView(centre, s)
        scene_view.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        rows = []
        for src in s.sources:
            row = SourceRow(left, src, scene_view)
            row.pack(fill="x", padx=4, pady=3)
            rows.append(row)

        scene_view.set_rows(rows)
        scene_view.after(100, scene_view.redraw)

        right = tk.Frame(body, bg=PANEL_BG, width=280)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        tk.Label(
            right, text="OUTPUT", bg=PANEL_BG, fg=ACCENT,
            font=("Helvetica", 9, "bold")
        ).pack(anchor="w", padx=10, pady=(10, 0))

        OutputPanel(right, s).pack(fill="both", expand=True)

        tk.Frame(body, bg=ACCENT2, width=1).place(in_=left, relx=1.0, rely=0, relheight=1)
        tk.Frame(body, bg=ACCENT2, width=1).place(in_=right, relx=0.0, rely=0, relheight=1)

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