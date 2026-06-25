"""
GUI application for the spatial audio pipeline.
Usage (from project root):
python src/gui/gui_app.py
"""

import threading
import tkinter as tk
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageTk

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
    ICONS_DIR,
    run_generate,
)
from gui_widgets import (
    SourceRow,
    SourceInspector,
    SceneView,
    OutputPanel,
    StatusBar,
    make_button_3d,
    _load_svg_icon,
    _make_stop_icon,
    _pil_rgb,
)


class SpatialAudioGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sonara - Ambisonic Render Engine")
        self.configure(bg=BG)
        self.geometry("1360x800")
        self.minsize(1160, 700)

        self.state = AppState()

        # Separate this process from python.exe in the Windows taskbar
        import ctypes, sys, tempfile, os
        if sys.platform == "win32":
            try:
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                    "Sonara.SpatialAudio"
                )
            except Exception:
                pass

        # Build a multi-size .ico from the 256-px PNG and set it as the window icon.
        # wm_iconbitmap(.ico) is the only reliable path to the Windows taskbar icon.
        try:
            _icon_src = Image.open("assets/mark-128.png").convert("RGBA")
            _ico_path = os.path.join(tempfile.gettempdir(), "sonara_app.ico")
            _ico_sizes = [16, 32, 48, 64, 128, 256]
            _ico_imgs = []
            for _s in _ico_sizes:
                if _s <= _icon_src.width:
                    # Downscale — LANCZOS is optimal
                    _ico_imgs.append(_icon_src.resize((_s, _s), Image.LANCZOS))
                else:
                    # Upscale — BICUBIC avoids LANCZOS ringing, then mild unsharp mask
                    _up = _icon_src.resize((_s, _s), Image.BICUBIC)
                    _up = _up.filter(ImageFilter.UnsharpMask(radius=0.8, percent=60, threshold=2))
                    _ico_imgs.append(_up)
            _ico_imgs[0].save(
                _ico_path, format="ICO", append_images=_ico_imgs[1:],
                sizes=[(s, s) for s in _ico_sizes],
            )
            self.wm_iconbitmap(_ico_path)
            self._app_icon_src = _icon_src  # prevent GC
        except Exception:
            pass
        self._generate_item = None
        self._title_item = None
        self._topbar_gradient = None
        self._play_item = None
        self._live_player = None
        self._scene_view = None
        self._build()

    def _make_generate_images(self):
        W, H = 160, 52
        ICON_SZ, GAP = 18, 8

        try:
            font = ImageFont.truetype("arialbd.ttf", 17)
        except OSError:
            font = ImageFont.load_default()

        try:
            icon = _load_svg_icon(str(ICONS_DIR / "generate.svg"), ICON_SZ, (255, 255, 255))
        except Exception:
            icon = None

        results = []
        for top_hex, bot_hex, pressed in [
            ("#D96820", "#B85010", False),
            ("#E87838", "#C85F28", False),
            ("#A84010", "#943810", True),
        ]:
            img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            r1, g1, b1 = _pil_rgb(top_hex)
            r2, g2, b2 = _pil_rgb(bot_hex)
            for y in range(H):
                t = y / max(1, H - 1)
                draw.line([(0, y), (W - 1, y)],
                          fill=(int(r1+(r2-r1)*t), int(g1+(g2-g1)*t), int(b1+(b2-b1)*t), 255))
            if not pressed:
                draw.line([(1, 1), (W - 2, 1)],
                          fill=(min(255, r1+55), min(255, g1+40), min(255, b1+25), 255))

            label = "Generate"
            bb = draw.textbbox((0, 0), label, font=font)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
            v = 1 if pressed else 0
            if icon is not None:
                total_w = ICON_SZ + GAP + tw
                ix = (W - total_w) // 2
                tx = ix + ICON_SZ + GAP - bb[0]
                img.paste(icon, (ix, (H - ICON_SZ) // 2 + v), icon)
            else:
                tx = (W - tw) // 2 - bb[0]
            ty = (H - th) // 2 - bb[1] + v
            draw.text((tx, ty), label, fill=(255, 255, 255, 255), font=font)
            results.append(ImageTk.PhotoImage(img))

        return results

    def _panel_header(self, parent, text):
        wrap = tk.Frame(parent, bg=PANEL_BG)
        wrap.pack(fill="x", padx=8, pady=(10, 4))

        tk.Frame(wrap, bg=ACCENT, width=3).pack(side="left", fill="y")
        tk.Label(
            wrap,
            text=text,
            bg=PANEL_BG,
            fg=TEXT,
            font=("Helvetica", 11, "bold"),
            anchor="w",
            padx=10,
            pady=6,
        ).pack(side="left")
        tk.Frame(wrap, bg=BORDER, height=1).pack(side="bottom", fill="x")

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
            "#0A0907",
            "#141210",
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

    def _render_play_images(self, text: str, disabled: bool = False):
        W, H = 150, 52
        ICON_SZ, GAP = 18, 8

        try:
            font = ImageFont.truetype("arialbd.ttf", 17)
        except OSError:
            font = ImageFont.load_default()

        icon = None
        if "Play" in text and not disabled:
            try:
                icon = _load_svg_icon(str(ICONS_DIR / "play.svg"), ICON_SZ, (255, 255, 255))
            except Exception:
                pass

        if disabled:
            states = [
                ("_play_img_normal",  "#7A4010", "#5A3008", False),
                ("_play_img_hover",   "#7A4010", "#5A3008", False),
                ("_play_img_pressed", "#7A4010", "#5A3008", False),
            ]
        else:
            states = [
                ("_play_img_normal",  "#D96820", "#B85010", False),
                ("_play_img_hover",   "#E87838", "#C85F28", False),
                ("_play_img_pressed", "#A84010", "#943810", True),
            ]

        for attr, top_hex, bot_hex, pressed in states:
            img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            r1, g1, b1 = _pil_rgb(top_hex)
            r2, g2, b2 = _pil_rgb(bot_hex)
            for y in range(H):
                t = y / max(1, H - 1)
                draw.line([(0, y), (W - 1, y)],
                          fill=(int(r1+(r2-r1)*t), int(g1+(g2-g1)*t), int(b1+(b2-b1)*t), 255))
            if not pressed and not disabled:
                draw.line([(1, 1), (W - 2, 1)],
                          fill=(min(255, r1+55), min(255, g1+40), min(255, b1+25), 255))

            bb = draw.textbbox((0, 0), text, font=font)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
            v = 1 if pressed else 0
            t_fill = (180, 160, 140, 200) if disabled else (255, 255, 255, 255)

            if icon is not None:
                total_w = ICON_SZ + GAP + tw
                ix = (W - total_w) // 2
                tx = ix + ICON_SZ + GAP - bb[0]
                img.paste(icon, (ix, (H - ICON_SZ) // 2 + v), icon)
            else:
                tx = (W - tw) // 2 - bb[0]
            ty = (H - th) // 2 - bb[1] + v
            draw.text((tx, ty), text, fill=t_fill, font=font)
            setattr(self, attr, ImageTk.PhotoImage(img))

        self._play_text = text
        self._play_disabled = disabled

    def _update_play_btn(self, text: str, disabled: bool = False):
        self._render_play_images(text, disabled)
        self._topbar.itemconfig(self._play_item, image=self._play_img_normal)

    def _on_play_enter(self, _event=None):
        if not self._play_disabled:
            self._topbar.itemconfig(self._play_item, image=self._play_img_hover)

    def _on_play_leave(self, _event=None):
        self._topbar.itemconfig(self._play_item, image=self._play_img_normal)

    def _on_play_press(self, _event=None):
        if not self._play_disabled:
            self._topbar.itemconfig(self._play_item, image=self._play_img_pressed)

    def _on_play_release(self, event):
        if self._play_disabled:
            return
        x = self._topbar.canvasx(event.x)
        y = self._topbar.canvasy(event.y)
        bbox = self._topbar.bbox(self._play_item)
        if bbox is None:
            return
        x1, y1, x2, y2 = bbox
        if x1 <= x <= x2 and y1 <= y <= y2:
            self._topbar.itemconfig(self._play_item, image=self._play_img_normal)
            self._on_play_stop()
        else:
            self._topbar.itemconfig(self._play_item, image=self._play_img_normal)

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
            bg="#0A0907",
            height=100,
            highlightthickness=0,
            bd=0,
        )
        self._topbar.pack(fill="x", side="top")
        self._topbar.bind("<Configure>", self._redraw_topbar)

        (self._gen_img_normal,
         self._gen_img_hover,
         self._gen_img_pressed) = self._make_generate_images()
        _logo_pil = Image.open("assets/title_logo.png").convert("RGBA")
        _bbox = _logo_pil.getbbox()  # crop transparent padding around the artwork
        if _bbox:
            _logo_pil = _logo_pil.crop(_bbox)
        _target_h = 57  # height of visible artwork in px — adjust to taste
        _target_w = round(_logo_pil.width * _target_h / _logo_pil.height)
        _logo_pil = _logo_pil.resize((_target_w, _target_h), Image.LANCZOS)
        self._title_img = ImageTk.PhotoImage(_logo_pil)

        self._title_item = self._topbar.create_image(0, 0, image=self._title_img, anchor="center")
        self._generate_item = self._topbar.create_image(0, 0, image=self._gen_img_normal, anchor="center")

        self._topbar.tag_bind(self._generate_item, "<Enter>", self._on_generate_enter)
        self._topbar.tag_bind(self._generate_item, "<Leave>", self._on_generate_leave)
        self._topbar.tag_bind(self._generate_item, "<ButtonPress-1>", self._on_generate_press)
        self._topbar.tag_bind(self._generate_item, "<ButtonRelease-1>", self._on_generate_release)

        self._play_text = "Play"
        self._play_disabled = False
        self._render_play_images("Play")
        self._play_item = self._topbar.create_image(0, 0, image=self._play_img_normal, anchor="center")
        self._topbar.tag_bind(self._play_item, "<Enter>", self._on_play_enter)
        self._topbar.tag_bind(self._play_item, "<Leave>", self._on_play_leave)
        self._topbar.tag_bind(self._play_item, "<ButtonPress-1>", self._on_play_press)
        self._topbar.tag_bind(self._play_item, "<ButtonRelease-1>", self._on_play_release)

        self._status = StatusBar(self)
        self._status.pack(fill="x", side="bottom")

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=0, minsize=360)
        body.grid_columnconfigure(1, weight=1, minsize=480)
        body.grid_columnconfigure(2, weight=0, minsize=380)

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
        self._scene_view = scene_view

        record_bar = tk.Frame(centre, bg=PANEL_BG)
        record_bar.pack(fill="x", padx=8, pady=(0, 8))

        record_btn = tk.Button(
            record_bar,
            text="Record Movement",
            font=("Helvetica", 10, "bold"),
            command=lambda: scene_view.toggle_recording(),
        )
        make_button_3d(
            record_btn,
            ACCENT2,
            fg=TEXT,
            border=BORDER,
            active_bg="#5A9048",
            pressed_bg="#3A6030",
        )
        try:
            _rec_col = _pil_rgb(TEXT)
            _rec_icon = _load_svg_icon(str(ICONS_DIR / "record.svg"), 13, _rec_col)
            _rec_photo = ImageTk.PhotoImage(_rec_icon)
            _stop_photo = ImageTk.PhotoImage(_make_stop_icon(13, _rec_col))
            record_btn.config(image=_rec_photo, compound="left", padx=6)
            record_btn._icon_ref = _rec_photo
            record_btn._record_photo = _rec_photo
            record_btn._stop_photo = _stop_photo
        except Exception:
            pass
        record_btn.pack(side="left")

        clear_btn = tk.Button(
            record_bar,
            text="Clear Movement",
            font=("Helvetica", 10, "bold"),
            command=lambda: self._on_clear_movement(scene_view),
        )
        make_button_3d(clear_btn, PANEL_BG2, active_bg=ACCENT, pressed_bg="#A8892E")
        try:
            _clr_icon = _load_svg_icon(str(ICONS_DIR / "clear.svg"), 13, _pil_rgb(TEXT))
            _clr_photo = ImageTk.PhotoImage(_clr_icon)
            clear_btn.config(image=_clr_photo, compound="left", padx=6)
            clear_btn._icon_ref = _clr_photo
        except Exception:
            pass
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
            if self._scene_view is not None:
                self._scene_view.set_live_player(None)
            self._update_play_btn("Play")
            return

        # Load HRTF + stems in a background thread to avoid freezing the UI
        self._update_play_btn("Loading…", disabled=True)

        def _start():
            try:
                from spatial_pipeline.live_player import LivePlayer
                from spatial_pipeline.config import DEFAULT_HRTF_SOFA

                sofa = str(self.state.hrtf_path) if self.state.hrtf_path else str(DEFAULT_HRTF_SOFA)
                player = LivePlayer(self.state, sofa)
                player.start()
                self._live_player = player
                self.after(0, lambda: self._update_play_btn("■  Stop"))
                if self._scene_view is not None:
                    self.after(0, lambda p=player: self._scene_view.set_live_player(p))
            except Exception as e:
                self._live_player = None
                self.after(0, lambda: self._update_play_btn("Play"))
                self.after(0, lambda: messagebox.showerror("Playback error", str(e)))

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