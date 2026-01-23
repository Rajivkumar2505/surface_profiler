import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import SAM, YOLO
from transformers import pipeline
import torch
import os
from datetime import datetime

class SurfaceProfileApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Surface Profile Analyzer")
        self.root.geometry("1400x900") # Widened for more buttons

        # --- State Variables for Folder Navigation ---
        self.directory_path = ""
        self.label_path = "" # New: Path for labels
        self.image_files = []
        self.current_index = -1

        # State Variables
        self.cv_img = None     
        self.pil_orig = None   
        self.tk_img = None
        self.img_id = None
        self.zoom_level = 1.0
        self.draw_mode = tk.StringVar(value="rect")
        
        self.points = []
        self.temp_shape = None  
        self.temp_line = None   
        self._after_id = None
        self.processed_mask = None # Ensure this is initialized

        # Models
        self.sam_model = SAM("sam2_b.pt")
        # self.sam_model = YOLO("yolo26n-seg.pt")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.depth_pipe = pipeline(task="depth-estimation", 
                                   model="depth-anything/Depth-Anything-V2-Small-hf", 
                                   device=self.device)

        self.setup_ui()

    def setup_ui(self):
        # --- Toolbar ---
        self.header = tk.Frame(self.root, bg="#2c3e50", pady=8)
        self.header.pack(side=tk.TOP, fill=tk.X)

        btn_style = {"bg": "#ecf0f1", "relief": tk.FLAT, "padx": 10}
        
        # Navigation
        tk.Button(self.header, text="Upload Image", command=self.upload_image, **btn_style).pack(side=tk.LEFT, padx=5)
        tk.Button(self.header, text="Open Folder", command=self.open_folder, **btn_style).pack(side=tk.LEFT, padx=5)
        tk.Button(self.header, text="◀ Prev", command=self.prev_image, **btn_style).pack(side=tk.LEFT, padx=2)
        tk.Button(self.header, text="Next ▶", command=self.next_image, **btn_style).pack(side=tk.LEFT, padx=2)
        
        # NEW Label Buttons
        tk.Button(self.header, text="Set Label Dir", command=self.set_label_dir, bg="#f39c12", fg="white", relief=tk.FLAT).pack(side=tk.LEFT, padx=5)
        tk.Button(self.header, text="Save Label", command=self.save_yolo_label, bg="#e67e22", fg="white", relief=tk.FLAT).pack(side=tk.LEFT, padx=5)

        radio_frame = tk.Frame(self.header, bg="#2c3e50")
        radio_frame.pack(side=tk.LEFT, padx=20)
        
        tk.Radiobutton(radio_frame, text="Rectangle", variable=self.draw_mode, value="rect", 
                       bg="#2c3e50", fg="white", selectcolor="#34495e", indicatoron=0, 
                       padx=10, command=self.reset_draw_state).pack(side=tk.LEFT)
        
        tk.Radiobutton(radio_frame, text="Polygon", variable=self.draw_mode, value="poly", 
                       bg="#2c3e50", fg="white", selectcolor="#34495e", indicatoron=0, 
                       padx=10, command=self.reset_draw_state).pack(side=tk.LEFT)

        tk.Button(self.header, text="Delete All", command=self.clear_canvas, **btn_style).pack(side=tk.LEFT, padx=5)
        tk.Button(self.header, text="Generate Mask", command=self.generate_mask, **btn_style).pack(side=tk.LEFT, padx=5)
        tk.Button(self.header, text="Save Mask", command=self.save_masked_crop, **btn_style).pack(side=tk.LEFT, padx=5)
        tk.Button(self.header, text="Surface Profile", command=self.view_3d, bg="#27ae60", fg="white", relief=tk.FLAT).pack(side=tk.LEFT, padx=10)
        tk.Button(self.header, text="✕", command=self.remove_image, bg="#c0392b", fg="white").pack(side=tk.RIGHT, padx=15)

        self.status_bar = tk.Label(self.root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W, padx=10)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas = tk.Canvas(self.root, bg="#121212", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bindings (re-using your existing logic)
        self.canvas.bind("<MouseWheel>", self.handle_zoom)
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_release)
        self.canvas.bind("<Motion>", self.on_mouse_move) 
        self.canvas.bind("<ButtonPress-3>", self.handle_right_press)
        self.canvas.bind("<B3-Motion>", self.do_pan)
        self.root.bind("<Escape>", lambda e: self.finish_polygon())

    def set_label_dir(self):
        folder = filedialog.askdirectory()
        if folder:
            self.label_path = folder
            self.status_bar.config(text=f"Label Directory: {self.label_path}")

    def save_yolo_label(self):
        if self.processed_mask is None:
            messagebox.showwarning("Warning", "Please generate a mask first.")
            return

        # 1. Determine Save Path
        current_img_name = "image"
        if self.current_index >= 0:
            current_img_name = os.path.splitext(self.image_files[self.current_index])[0]
        
        save_dir = self.label_path if self.label_path else (self.directory_path if self.directory_path else os.getcwd())
        txt_path = os.path.join(save_dir, f"{current_img_name}.txt")

        # 2. Extract Contours for YOLO Segmentation Format
        # YOLO format for segmentation: <class-index> <x1> <y1> <x2> <y2> ...
        contours, _ = cv2.findContours(self.processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = self.processed_mask.shape
        yolo_lines = []
        
        for cnt in contours:
            # We filter small noise
            if cv2.contourArea(cnt) < 10: continue
            
            # Normalize coordinates between 0 and 1
            points = cnt.flatten().tolist()
            normalized_points = []
            for i in range(0, len(points), 2):
                normalized_points.append(str(round(points[i] / w, 6)))     # x
                normalized_points.append(str(round(points[i+1] / h, 6)))   # y
            
            # Assuming class index 0 for 'surface'
            yolo_lines.append(f"0 {' '.join(normalized_points)}")

        # 3. Save to file
        try:
            with open(txt_path, "w") as f:
                f.write("\n".join(yolo_lines))
            messagebox.showinfo("Success", f"YOLO label saved to:\n{txt_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save label: {e}")

    # --- NEW NAVIGATION METHODS ---
    def open_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.directory_path = folder_selected
            # Filter for common image extensions
            extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
            self.image_files = [f for f in os.listdir(self.directory_path) if f.lower().endswith(extensions)]
            self.image_files.sort() # Sort alphabetically
            
            if self.image_files:
                self.current_index = 0
                self.load_image_from_list()
            else:
                messagebox.showwarning("Warning", "No image files found in the selected folder.")

    def load_image_from_list(self):
        if 0 <= self.current_index < len(self.image_files):
            full_path = os.path.join(self.directory_path, self.image_files[self.current_index])
            self.clear_canvas() # Clear old masks/shapes when switching images
            self.load_image_logic(full_path)
            self.root.title(f"Analyzer - {self.image_files[self.current_index]} ({self.current_index + 1}/{len(self.image_files)})")

    def next_image(self):
        if not self.image_files: return
        self.current_index = (self.current_index + 1) % len(self.image_files)
        self.load_image_from_list()

    def prev_image(self):
        if not self.image_files: return
        self.current_index = (self.current_index - 1) % len(self.image_files)
        self.load_image_from_list()

    # --- UPDATED LOGIC TO HANDLE BOTH UPLOAD AND FOLDER ---
    def upload_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.image_files = [] # Reset folder list if single image is uploaded
            self.load_image_logic(path)

    def load_image_logic(self, path):
        """Internal method to process the image file path."""
        if path:
            self.pil_orig = Image.open(path).convert("RGB")
            self.cv_img = np.array(self.pil_orig)
            
            self.root.update_idletasks()
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()
            img_w, img_h = self.pil_orig.size

            self.zoom_level = min(canvas_w / img_w, canvas_h / img_h)
            self.render_image(fast=False)
            self.canvas.xview_moveto(0)
            self.canvas.yview_moveto(0)
            self.canvas.focus_set()

    # --- REST OF THE METHODS REMAIN UNCHANGED ---
    def reset_draw_state(self):
        self.points = []
        self.canvas.delete("temp")

    def render_image(self, fast=True):
            if self.pil_orig is None: return
            w, h = self.pil_orig.size
            new_size = (max(1, int(w * self.zoom_level)), max(1, int(h * self.zoom_level)))
            resample = Image.Resampling.NEAREST if fast else Image.Resampling.LANCZOS
            resized = self.pil_orig.resize(new_size, resample)
            self.tk_img = ImageTk.PhotoImage(resized)
            if self.img_id:
                self.canvas.itemconfig(self.img_id, image=self.tk_img)
            else:
                self.img_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img, tags="main_img")
            self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def handle_zoom(self, event):
        if not self.pil_orig: return
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        if event.num == 4 or event.delta > 0:
            factor = 1.1
        else:
            factor = 0.9
        old_zoom = self.zoom_level
        self.zoom_level = max(0.05, min(self.zoom_level * factor, 20.0))
        self.render_image(fast=True)
        self.canvas.scan_mark(event.x, event.y)
        self.canvas.scan_dragto(int(x * (self.zoom_level / old_zoom)), 
                                int(y * (self.zoom_level / old_zoom)), gain=1)
        if self._after_id: self.root.after_cancel(self._after_id)
        self._after_id = self.root.after(100, lambda: self.render_image(fast=False))

    def on_left_click(self, event):
            self.canvas.focus_set()
            if not self.img_id: return
            if self.draw_mode.get() == "rect":
                self.clear_previous_analysis() 
            x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
            if self.draw_mode.get() == "rect":
                self.points = [(x, y)]
            else:
                self.points.append((x, y))
                if len(self.points) > 1:
                    self.canvas.create_line(self.points[-2], self.points[-1], 
                                            fill="cyan", width=2, tags="shape")
                    
    def clear_previous_analysis(self):
            self.canvas.delete("shape", "temp")
            self.points = []
            self.processed_mask = None
            self.masked_image = None
            if self.pil_orig:
                self.render_image(fast=False)
                    
    def on_mouse_drag(self, event):
        if not self.img_id or not self.points: return
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        if self.draw_mode.get() == "rect":
            if self.temp_shape: self.canvas.delete(self.temp_shape)
            x1, y1 = self.points[0]
            self.temp_shape = self.canvas.create_rectangle(x1, y1, x, y, outline="lime", width=2, tags="temp")

    def on_left_release(self, event):
            if self.draw_mode.get() == "rect" and self.points:
                x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
                if self.temp_shape: 
                    self.canvas.delete(self.temp_shape)
                    self.temp_shape = None
                x1, y1 = self.points[0]
                if abs(x - x1) > 5 and abs(y - y1) > 5:
                    self.canvas.create_rectangle(x1, y1, x, y, outline="lime", width=2, tags="shape")
                self.points = []

    def on_mouse_move(self, event):
        if self.pil_orig:
            x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
            ix, iy = int(x / self.zoom_level), int(y / self.zoom_level)
            w, h = self.pil_orig.size
            if 0 <= ix < w and 0 <= iy < h:
                p = self.cv_img[iy, ix]
                self.status_bar.config(text=f"Pixel: ({ix}, {iy}) | RGB: {tuple(p)}")
        if self.draw_mode.get() == "poly" and self.points:
            if self.temp_line: self.canvas.delete(self.temp_line)
            x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
            x1, y1 = self.points[-1]
            self.temp_line = self.canvas.create_line(x1, y1, x, y, fill="cyan", dash=(4, 4), tags="temp")

    def handle_right_press(self, event):
        if self.draw_mode.get() == "poly" and len(self.points) > 1:
            self.finish_polygon()
        else:
            self.canvas.scan_mark(event.x, event.y)

    def do_pan(self, event):
        if self.draw_mode.get() == "poly" and self.points: return
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def finish_polygon(self):
        if self.draw_mode.get() == "poly" and len(self.points) > 2:
            if self.temp_line: self.canvas.delete(self.temp_line)
            self.canvas.create_line(self.points[-1], self.points[0], fill="cyan", width=2, tags="shape")
            self.points = []

    def clear_canvas(self):
            self.canvas.delete("shape", "temp")
            self.points = []
            self.processed_mask = None
            self.masked_image = None
            if self.pil_orig:
                self.render_image(fast=False)
            self.status_bar.config(text="All annotations and masks cleared.")

    def remove_image(self):
        self.canvas.delete("all")
        self.img_id = None
        self.pil_orig = None
        self.cv_img = None
        self.processed_mask = None
        self.masked_image = None
        self.zoom_level = 1.0
        self.image_files = []
        self.status_bar.config(text="Image and data removed.")

    def generate_mask(self):
            if self.cv_img is None or not self.canvas.find_withtag("shape"):
                messagebox.showwarning("Warning", "Please upload an image and draw a rectangle first.")
                return
            all_shapes = self.canvas.find_withtag("shape")
            bbox_id = None
            for s in reversed(all_shapes):
                if self.canvas.type(s) == "rectangle":
                    bbox_id = s
                    break
            if bbox_id is None:
                messagebox.showwarning("Warning", "No rectangle found.")
                return
            c_coords = self.canvas.coords(bbox_id) 
            x1, y1 = int(c_coords[0] / self.zoom_level), int(c_coords[1] / self.zoom_level)
            x2, y2 = int(c_coords[2] / self.zoom_level), int(c_coords[3] / self.zoom_level)
            roi_bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            try:
                results = self.sam_model(self.cv_img, bboxes=[roi_bbox])
                if results[0].masks is None:
                    messagebox.showerror("Error", "SAM detected no objects.")
                    return
                mask = results[0].masks.data[0].cpu().numpy()
                mask = (mask > 0).astype(np.uint8)
                if mask.shape[:2] != self.cv_img.shape[:2]:
                    mask = cv2.resize(mask, (self.cv_img.shape[1], self.cv_img.shape[0]))
                self.processed_mask = mask
                overlay = self.cv_img.copy()
                overlay[mask == 1] = [0, 255, 0] 
                alpha = 0.5
                blended_img = cv2.addWeighted(overlay, alpha, self.cv_img, 1 - alpha, 0)
                self.display_pil = Image.fromarray(blended_img)
                self.render_masked_view()
            except Exception as e:
                messagebox.showerror("Model Error", f"SAM failed: {str(e)}")

    def render_masked_view(self):
        w, h = self.display_pil.size
        new_size = (max(1, int(w * self.zoom_level)), max(1, int(h * self.zoom_level)))
        resized = self.display_pil.resize(new_size, Image.Resampling.NEAREST)
        self.tk_img = ImageTk.PhotoImage(resized)
        self.canvas.itemconfig(self.img_id, image=self.tk_img)

    def view_3d(self):
        if self.processed_mask is None:
            messagebox.showwarning("Warning", "Please generate a mask first.")
            return
        try:
            all_shapes = self.canvas.find_withtag("shape")
            bbox_id = next((s for s in reversed(all_shapes) if self.canvas.type(s) == "rectangle"), None)
            if not bbox_id: return
            c_coords = self.canvas.coords(bbox_id)
            img_bbox = self.canvas.bbox(self.img_id)
            x1 = int((min(c_coords[0], c_coords[2]) - img_bbox[0]) / self.zoom_level)
            y1 = int((min(c_coords[1], c_coords[3]) - img_bbox[1]) / self.zoom_level)
            x2 = int((max(c_coords[0], c_coords[2]) - img_bbox[0]) / self.zoom_level)
            y2 = int((max(c_coords[1], c_coords[3]) - img_bbox[1]) / self.zoom_level)
            roi_img = self.cv_img[y1:y2, x1:x2]
            roi_mask = self.processed_mask[y1:y2, x1:x2]
            roi_img = self.run_depth_analysis(roi_img)
            masked_depth = (roi_img * roi_mask).astype(np.float32)
            h, w = masked_depth.shape
            step = 1
            X, Y = np.meshgrid(np.arange(0, w, step), np.arange(0, h, step))
            Z = masked_depth[::step, ::step]
            Z[Z < 1] = np.nan 
            fig = plt.figure(figsize=(12, 6))
            fig.canvas.manager.set_window_title('Surface Profile')
            ax1 = fig.add_subplot(121)
            ax1.set_title("2D Surface profile")
            Zm = np.ma.masked_invalid(Z)
            cp = ax1.contourf(X, Y, Zm, cmap="plasma", levels=20)
            ax1.set_box_aspect(1) 
            ax2 = fig.add_subplot(122, projection='3d')
            ax2.set_title("3D Surface Profile")
            surf = ax2.plot_surface(X, Y, Z, cmap='plasma', edgecolor='none', antialiased=True)
            ax2.contourf(X, Y, Z, zdir="z", offset=0, cmap="plasma", alpha=0.5)
            ax2.grid(False)
            ax2.invert_yaxis() 
            fig.colorbar(surf, ax=ax2, shrink=0.6, aspect=10)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            messagebox.showerror("3D Error", f"Failed: {str(e)}")

    def run_depth_analysis(self, cropped_img_np):
            pil_img = Image.fromarray(cropped_img_np.astype('uint8'))
            res = self.depth_pipe(pil_img)
            return np.array(res["depth"])

    def save_masked_crop(self):
        if self.processed_mask is None or self.cv_img is None:
            messagebox.showwarning("Warning", "Please generate a mask first.")
            return
        all_shapes = self.canvas.find_withtag("shape")
        bbox_id = next((s for s in reversed(all_shapes) if self.canvas.type(s) == "rectangle"), None)
        if not bbox_id: return
        c_coords = self.canvas.coords(bbox_id)
        x1 = int(min(c_coords[0], c_coords[2]) / self.zoom_level)
        y1 = int(min(c_coords[1], c_coords[3]) / self.zoom_level)
        x2 = int(max(c_coords[0], c_coords[2]) / self.zoom_level)
        y2 = int(max(c_coords[1], c_coords[3]) / self.zoom_level)
        mask_3d = np.repeat(self.processed_mask[:, :, np.newaxis], 3, axis=2)
        masked_full = self.cv_img * mask_3d
        cropped_masked = masked_full[y1:y2, x1:x2]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mask_{timestamp}.jpg"
        save_path = os.path.join(os.getcwd(), filename)
        try:
            output_img = cv2.cvtColor(cropped_masked, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, output_img)
            messagebox.showinfo("Success", f"Masked crop saved automatically to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save image: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SurfaceProfileApp(root)
    root.mainloop()