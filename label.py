import argparse
import math
import os
import queue
import threading
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

from label_store import LabelStore

# ── visual constants ──────────────────────────────────────────────────────────

PADDING = 5
CELL_BG = '#f0f0f0'
SELECTED_BG = '#b8d4f8'
LABEL_COLORS = {
    'ad':      '#cc2222',
    'content': '#007700',
    'neither': '#0055cc',
    'ignore':  '#888888',
    None:      '#888888',
}


# ── thumbnail cell ────────────────────────────────────────────────────────────

class ThumbnailCell:
    """One cell in the grid: border frame + image + filename/label text."""

    def __init__(self, parent, row, thumb_size, selected, on_click, on_double_click):
        self.filename = row['filename']
        bg = SELECTED_BG if selected else CELL_BG

        # Outer frame — background shows as a colored border via padx/pady
        self.frame = tk.Frame(parent, bg=bg, padx=3, pady=3, cursor='hand2')

        self.img_label = tk.Label(
            self.frame, bg=bg,
            width=thumb_size, height=thumb_size,
        )
        self.img_label.pack()

        short = os.path.basename(row['filename'])
        if len(short) > 18:
            short = '…' + short[-17:]
        body = short
        if row['label']:
            body += '\n' + row['label'].replace('_', ' ')
        fg = LABEL_COLORS.get(row['label'], LABEL_COLORS[None])
        self.text_label = tk.Label(
            self.frame, text=body, fg=fg, bg=bg,
            font=('Helvetica', 9), wraplength=thumb_size, justify='center',
        )
        self.text_label.pack()

        for w in (self.frame, self.img_label, self.text_label):
            w.bind('<Button-1>', on_click)
            w.bind('<Double-Button-1>', lambda e, cb=on_double_click: cb())

    def set_image(self, photo):
        self.img_label.config(image=photo)

    def set_selected(self, selected):
        bg = SELECTED_BG if selected else CELL_BG
        self.frame.config(bg=bg)
        self.img_label.config(bg=bg)
        self.text_label.config(bg=bg)


# ── main application ──────────────────────────────────────────────────────────

class LabelApp:
    def __init__(self, root, args):
        self.root = root
        self.root.title('Hush Labeler')
        self.store = LabelStore(args.db)
        self.thumb_size = args.thumb_size
        self.page_size = args.page_size

        self._all_rows = []
        self._filtered_rows = []
        self._page = 0
        self._selected = set()          # set of filenames selected on current page
        self._last_clicked_idx = None   # page-local index of last click
        self._undo_stack = []           # list of (filenames, {filename: old_label})
        self._cells = []                # ThumbnailCell list for current page
        self._photo_refs = []           # PhotoImage refs (GC prevention)
        self._load_stop_event = threading.Event()
        self._resize_after_id = None
        self._last_width = 0
        self._effective_page_size = args.page_size  # updated each render from window height

        self._build_ui()
        self._reload()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # Top controls
        top = tk.Frame(self.root)
        top.pack(side='top', fill='x', padx=6, pady=4)

        tk.Label(top, text='Filter:').pack(side='left')
        self._filter_var = tk.StringVar(value='All')
        cb = ttk.Combobox(
            top, textvariable=self._filter_var,
            values=['All', 'Unlabeled', 'Ad', 'Content', 'Neither', 'Ignore'],
            state='readonly', width=9,
        )
        cb.pack(side='left', padx=(2, 8))
        cb.bind('<<ComboboxSelected>>', lambda e: self._apply_filter())

        tk.Label(top, text='Search:').pack(side='left')
        self._search_var = tk.StringVar()
        self._search_entry = tk.Entry(top, textvariable=self._search_var, width=22)
        self._search_entry.pack(side='left', padx=2)
        self._search_entry.bind('<Return>', lambda e: self._apply_filter())
        tk.Button(top, text='Go', command=self._apply_filter).pack(side='left')
        tk.Button(top, text='✕', command=self._clear_search, width=2).pack(side='left', padx=(2, 10))

        tk.Button(top, text='Reload', command=self._reload).pack(side='left')
        tk.Button(top, text='Help ?', command=self._show_help).pack(side='left', padx=(6, 0))

        # Pagination (right side, packed right-to-left)
        self._status_var = tk.StringVar()
        tk.Label(top, textvariable=self._status_var).pack(side='right', padx=10)
        self._btn_next = tk.Button(top, text='Next ›', command=self._next_page)
        self._btn_next.pack(side='right')
        self._btn_prev = tk.Button(top, text='‹ Prev', command=self._prev_page)
        self._btn_prev.pack(side='right', padx=(0, 4))

        # Legend — pack before grid so it reserves space before expand=True claims the rest
        legend = tk.Frame(self.root, bg='#e0e0e0')
        legend.pack(side='bottom', fill='x')
        tk.Label(
            legend,
            text=(
                'Click: select  ·  Shift+click: range  ·  Cmd+A: select all  ·  A: ad  ·  C: content  ·'
                '  N: neither  ·  X: ignore  ·  D: delete file  ·  Z/dbl-click: zoom'
                '  ·  Ctrl+Z: undo  ·  Esc: clear  ·  ←/→: page'
            ),
            bg='#e0e0e0', fg='#555555', font=('Helvetica', 10),
        ).pack(side='left', padx=8, pady=4)

        # Grid
        self._grid_frame = tk.Frame(self.root, bg=CELL_BG)
        self._grid_frame.pack(side='top', fill='both', expand=True, padx=6)

        # Key bindings on root window
        self.root.bind('<Command-a>', self._on_key_select_all)
        self.root.bind('<Key-a>', self._on_key_a)
        self.root.bind('<Key-c>', self._on_key_c)
        self.root.bind('<Key-n>', self._on_key_n)
        self.root.bind('<Key-x>', self._on_key_x)
        self.root.bind('<Key-d>', self._on_key_d)
        self.root.bind('<Key-z>', self._on_key_z)
        self.root.bind('<Escape>', lambda e: self._clear_selection())
        self.root.bind('<Control-z>', lambda e: self._undo())
        self.root.bind('<Left>', self._on_key_left)
        self.root.bind('<Right>', self._on_key_right)
        self.root.bind('<Configure>', self._on_resize)

    # ── help ──────────────────────────────────────────────────────────────────

    def _show_help(self):
        win = tk.Toplevel(self.root)
        win.title('Hush Labeler — Help')
        win.resizable(False, False)
        text = (
            'LABELING\n'
            '  A          Label selected as Ad\n'
            '  C          Label selected as Content (game)\n'
            '  N          Label selected as Neither (tabbed away, etc.)\n'
            '  X          Mark selected as Ignore (corrupted / noisy frame)\n'
            '  D          Permanently delete selected files from disk\n'
            '  Ctrl+Z     Undo last label action  (delete cannot be undone)\n'
            '\n'
            'SELECTION\n'
            '  Click        Toggle selection on a thumbnail\n'
            '  Shift+click  Extend selection to a range\n'
            '  Cmd+A        Select all on current page\n'
            '  Escape       Clear selection\n'
            '\n'
            'NAVIGATION\n'
            '  ← / →      Previous / next page\n'
            '  Filter     Drop-down to show All / Unlabeled / Ad / Content / Neither / Ignore\n'
            '  Search     Type a partial filename and press Go (or Enter)\n'
            '  Reload     Re-read DB and filesystem (pick up new captures)\n'
            '\n'
            'ZOOM\n'
            '  Z          Open zoom window for last-clicked thumbnail\n'
            '  Dbl-click  Open zoom window\n'
            '  ← / →      Navigate images while in zoom window\n'
            '  A/C/N/X    Relabel current image and close zoom\n'
            '  D          Delete current image and advance to next\n'
            '  Escape     Close zoom window\n'
        )
        tk.Label(
            win, text=text, justify='left', font=('Courier', 12),
            padx=20, pady=16, bg='white',
        ).pack()
        tk.Button(win, text='Close', command=win.destroy).pack(pady=(0, 12))
        win.focus_set()

    # ── data / filter ─────────────────────────────────────────────────────────

    def _reload(self):
        all_rows = self.store.get_all()
        # Auto-purge DB records whose files have been deleted from disk.
        missing = [r['filename'] for r in all_rows if not os.path.isfile(r['filename'])]
        if missing:
            self.store.delete_screenshots(missing)
            all_rows = [r for r in all_rows if r['filename'] not in set(missing)]
        self._all_rows = all_rows
        self._apply_filter()

    def _apply_filter(self):
        fval = self._filter_var.get()
        search = self._search_var.get().strip()

        rows = self._all_rows
        if fval == 'Unlabeled':
            rows = [r for r in rows if r['label'] is None]
        elif fval == 'Ad':
            rows = [r for r in rows if r['label'] == 'ad']
        elif fval == 'Content':
            rows = [r for r in rows if r['label'] == 'content']
        elif fval == 'Neither':
            rows = [r for r in rows if r['label'] == 'neither']
        elif fval == 'Ignore':
            rows = [r for r in rows if r['label'] == 'ignore']
        if search:
            rows = [r for r in rows if search in r['filename']]

        self._filtered_rows = rows
        self._page = 0
        self._selected.clear()
        self._last_clicked_idx = None
        self._render_page()

    def _clear_search(self):
        self._search_var.set('')
        self._apply_filter()

    # ── pagination ────────────────────────────────────────────────────────────

    def _total_pages(self):
        return max(1, math.ceil(len(self._filtered_rows) / self._effective_page_size))

    def _current_page_rows(self):
        start = self._page * self._effective_page_size
        return self._filtered_rows[start:start + self._effective_page_size]

    def _next_page(self):
        if self._page < self._total_pages() - 1:
            self._page += 1
            self._selected.clear()
            self._last_clicked_idx = None
            self._render_page()

    def _prev_page(self):
        if self._page > 0:
            self._page -= 1
            self._selected.clear()
            self._last_clicked_idx = None
            self._render_page()

    def _update_status(self):
        unlabeled = sum(1 for r in self._all_rows if r['label'] is None)
        self._status_var.set(
            f'Page {self._page + 1}/{self._total_pages()}  ·  '
            f'{len(self._filtered_rows)} shown  ·  '
            f'{unlabeled}/{len(self._all_rows)} unlabeled'
        )
        self._btn_prev.config(state='normal' if self._page > 0 else 'disabled')
        self._btn_next.config(
            state='normal' if self._page < self._total_pages() - 1 else 'disabled'
        )

    # ── rendering ─────────────────────────────────────────────────────────────

    def _render_page(self):
        # Cancel any in-progress image load
        self._load_stop_event.set()
        self._load_stop_event = threading.Event()

        # Compute layout dimensions first so page size is correct before we slice rows.
        self.root.update_idletasks()
        avail_w = max(self._grid_frame.winfo_width(), 900)
        avail_h = max(self._grid_frame.winfo_height(), 200)
        cell_w = self.thumb_size + PADDING * 2 + 10   # 10 for border + spacing
        cell_h = self.thumb_size + 50                  # thumb + frame padding + text label
        n_cols = max(1, avail_w // cell_w)
        n_rows = max(1, avail_h // cell_h)
        self._effective_page_size = n_cols * n_rows
        # Clamp page index in case effective size shrank (e.g. window made smaller).
        self._page = min(self._page, self._total_pages() - 1)

        for w in self._grid_frame.winfo_children():
            w.destroy()
        self._cells = []
        self._photo_refs = []
        self._last_clicked_idx = None

        rows = self._current_page_rows()
        self._update_status()

        if not rows:
            tk.Label(
                self._grid_frame,
                text='No images to display.',
                bg=CELL_BG, fg='#888888', font=('Helvetica', 14),
            ).pack(pady=60)
            return

        for i, row in enumerate(rows):
            r_idx, c_idx = divmod(i, n_cols)
            cell = ThumbnailCell(
                self._grid_frame, row, self.thumb_size,
                selected=(row['filename'] in self._selected),
                on_click=lambda event, idx=i: self._on_click(idx, event),
                on_double_click=lambda idx=i: self._open_zoom(idx),
            )
            cell.frame.grid(
                row=r_idx, column=c_idx,
                padx=PADDING // 2, pady=PADDING // 2,
                sticky='n',
            )
            self._cells.append(cell)
            self._photo_refs.append(None)

        self._start_load(rows)

    def _on_resize(self, event):
        if event.widget is not self.root:
            return
        if abs(event.width - self._last_width) > 80:
            self._last_width = event.width
            if self._resize_after_id:
                self.root.after_cancel(self._resize_after_id)
            self._resize_after_id = self.root.after(250, self._render_page)

    # ── background image loading ──────────────────────────────────────────────

    def _start_load(self, rows):
        stop = self._load_stop_event
        thumb_size = self.thumb_size
        q = queue.Queue()

        def worker():
            for i, row in enumerate(rows):
                if stop.is_set():
                    return
                try:
                    img = Image.open(row['filename']).convert('RGB')
                    img.thumbnail((thumb_size, thumb_size), Image.LANCZOS)
                    q.put((i, img))
                except Exception:
                    q.put((i, None))
            q.put(None)  # sentinel: done

        threading.Thread(target=worker, daemon=True).start()

        def poll():
            if stop.is_set():
                return
            done = False
            for _ in range(10):     # drain up to 10 items per tick
                try:
                    item = q.get_nowait()
                    if item is None:
                        done = True
                        break
                    i, pil_img = item
                    if pil_img is not None and i < len(self._cells):
                        photo = ImageTk.PhotoImage(pil_img)
                        self._photo_refs[i] = photo   # must hold reference
                        self._cells[i].set_image(photo)
                except queue.Empty:
                    break
            if not done:
                self.root.after(50, poll)

        self.root.after(50, poll)

    # ── click / selection ─────────────────────────────────────────────────────

    def _on_click(self, idx, event):
        self.root.focus_set()  # reclaim focus from filter dropdown or search box
        rows = self._current_page_rows()
        if idx >= len(rows):
            return
        fname = rows[idx]['filename']
        if event.state & 0x1:  # Shift held
            if self._last_clicked_idx is not None:
                lo = min(self._last_clicked_idx, idx)
                hi = max(self._last_clicked_idx, idx)
                for j in range(lo, hi + 1):
                    self._selected.add(rows[j]['filename'])
            else:
                self._selected.add(fname)
        else:
            if fname in self._selected:
                self._selected.discard(fname)
            else:
                self._selected.add(fname)
        self._last_clicked_idx = idx
        self._refresh_selection_visuals()

    def _clear_selection(self):
        self._selected.clear()
        self._refresh_selection_visuals()

    def _refresh_selection_visuals(self):
        rows = self._current_page_rows()
        for i, cell in enumerate(self._cells):
            if i < len(rows):
                cell.set_selected(rows[i]['filename'] in self._selected)

    # ── labeling ──────────────────────────────────────────────────────────────

    def _label_selected(self, label):
        if not self._selected:
            return
        filenames = list(self._selected)
        fname_set = set(filenames)
        # Record old labels for undo
        old_labels = {
            r['filename']: r['label']
            for r in self._all_rows
            if r['filename'] in fname_set
        }
        self._undo_stack.append((filenames, old_labels))
        self.store.set_labels_bulk(filenames, label)
        self._selected.clear()
        # Update in-memory rows directly — avoids a full DB re-query + file-existence scan.
        for r in self._all_rows:
            if r['filename'] in fname_set:
                r['label'] = label
        self._apply_filter()

    def _delete_selected(self):
        if not self._selected:
            return
        filenames = list(self._selected)
        for f in filenames:
            try:
                os.remove(f)
            except FileNotFoundError:
                pass
        self.store.delete_screenshots(filenames)
        self._selected.clear()
        # Purge from in-memory list immediately so _apply_filter sees clean state.
        deleted = set(filenames)
        self._all_rows = [r for r in self._all_rows if r['filename'] not in deleted]
        self._apply_filter()

    def _undo(self):
        if not self._undo_stack:
            return
        filenames, old_labels = self._undo_stack.pop()
        to_clear = [f for f, lbl in old_labels.items() if lbl is None]
        if to_clear:
            self.store.clear_labels_bulk(to_clear)
        for lbl in ('ad', 'content', 'neither', 'ignore'):
            subset = [f for f, l in old_labels.items() if l == lbl]
            if subset:
                self.store.set_labels_bulk(subset, lbl)
        self._reload()

    # ── zoom window ───────────────────────────────────────────────────────────

    def _open_zoom(self, idx):
        rows = self._current_page_rows()
        if idx >= len(rows):
            return

        win = tk.Toplevel(self.root)
        win.resizable(True, True)

        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        max_w = int(sw * 0.90)
        max_h = int(sh * 0.82)

        state = {'idx': idx}

        img_label = tk.Label(win)
        img_label.pack()
        info_label = tk.Label(win, font=('Helvetica', 11), pady=4)
        info_label.pack()

        def show(i):
            current_rows = self._current_page_rows()
            if i < 0 or i >= len(current_rows):
                return
            state['idx'] = i
            row = current_rows[i]
            try:
                img = Image.open(row['filename']).convert('RGB')
                img.thumbnail((max_w, max_h), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                img_label.config(image=photo)
                img_label.image = photo     # GC anchor on the label itself
            except Exception as exc:
                img_label.config(image='', text=f'Cannot load image:\n{exc}')
            lbl = row['label'] or 'unlabeled'
            info_label.config(
                text=f'{os.path.basename(row["filename"])}  [{lbl}]',
                fg=LABEL_COLORS.get(row['label'], LABEL_COLORS[None]),
            )
            win.title(f'{os.path.basename(row["filename"])}  [{lbl}]')

        def relabel_and_close(lbl):
            current_rows = self._current_page_rows()
            row = current_rows[state['idx']]
            self.store.set_label(row['filename'], lbl)
            win.destroy()
            self._reload()

        def delete_and_advance():
            current_rows = self._current_page_rows()
            row = current_rows[state['idx']]
            fname = row['filename']
            try:
                os.remove(fname)
            except FileNotFoundError:
                pass
            self.store.delete_screenshots([fname])
            deleted = {fname}
            self._all_rows = [r for r in self._all_rows if r['filename'] not in deleted]
            self._apply_filter()
            # Try to stay in zoom at the same index (next image slides in), else close.
            new_rows = self._current_page_rows()
            next_idx = min(state['idx'], len(new_rows) - 1)
            if next_idx >= 0:
                show(next_idx)
            else:
                win.destroy()

        show(idx)

        win.bind('<Left>', lambda e: show(state['idx'] - 1))
        win.bind('<Right>', lambda e: show(state['idx'] + 1))
        win.bind('<Key-a>', lambda e: relabel_and_close('ad'))
        win.bind('<Key-c>', lambda e: relabel_and_close('content'))
        win.bind('<Key-n>', lambda e: relabel_and_close('neither'))
        win.bind('<Key-x>', lambda e: relabel_and_close('ignore'))
        win.bind('<Key-d>', lambda e: delete_and_advance())
        win.bind('<Escape>', lambda e: win.destroy())
        win.focus_set()

    # ── key event routing ─────────────────────────────────────────────────────

    def _text_widget_focused(self):
        """True when a text-input widget has focus (suppress hotkeys)."""
        cls = getattr(self.root.focus_get(), 'winfo_class', lambda: '')()
        return cls in ('Entry', 'TCombobox', 'Text')

    def _on_key_select_all(self, event):
        if not self._text_widget_focused():
            for row in self._current_page_rows():
                self._selected.add(row['filename'])
            self._refresh_selection_visuals()

    def _on_key_a(self, event):
        if not self._text_widget_focused():
            self._label_selected('ad')

    def _on_key_c(self, event):
        if not self._text_widget_focused():
            self._label_selected('content')

    def _on_key_n(self, event):
        if not self._text_widget_focused():
            self._label_selected('neither')

    def _on_key_x(self, event):
        if not self._text_widget_focused():
            self._label_selected('ignore')

    def _on_key_d(self, event):
        if not self._text_widget_focused():
            self._delete_selected()

    def _on_key_z(self, event):
        if not self._text_widget_focused() and self._last_clicked_idx is not None:
            self._open_zoom(self._last_clicked_idx)

    def _on_key_left(self, event):
        if not self._text_widget_focused():
            self._prev_page()

    def _on_key_right(self, event):
        if not self._text_widget_focused():
            self._next_page()


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Label screenshots for hush ad detection.'
    )
    parser.add_argument('--db', default='hush.db',
                        help='SQLite database path (default: hush.db)')
    parser.add_argument('--screenshot-dir', default='screenshots',
                        help='Screenshot directory (default: screenshots)')
    parser.add_argument('--thumb-size', type=int, default=150,
                        help='Thumbnail size in pixels (default: 150)')
    parser.add_argument('--page-size', type=int, default=48,
                        help='Thumbnails per page (default: 48)')
    args = parser.parse_args()

    root = tk.Tk()
    root.geometry('1200x850')
    LabelApp(root, args)
    root.mainloop()


if __name__ == '__main__':
    main()
