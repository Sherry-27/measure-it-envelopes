"""Small local photo picker for the assessment demonstration."""
import argparse
import json
import queue
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


class Desktop:
    def __init__(self, root, output):
        self.root=root
        self.repo=Path(__file__).resolve().parents[1]
        self.output=output.resolve()
        self.events=queue.Queue()
        self.meter=None
        self.photo=None
        self.last_overlay=None
        self.image_ref=None
        root.title("Measure It - Envelope measurement")
        root.geometry("1000x790")
        root.minsize(800,650)
        style=ttk.Style(root)
        style.theme_use("clam")
        style.configure("TButton",padding=(12,9))
        outer=ttk.Frame(root,padding=18)
        outer.pack(fill="both",expand=True)
        ttk.Label(outer,text="Measure It",font=("Segoe UI",22,"bold")).pack(anchor="w")
        ttk.Label(outer,text="Choose an original iPhone photo with the envelope and the 36 mm marker visible.",
                  font=("Segoe UI",10)).pack(anchor="w",pady=(3,12))
        bar=ttk.Frame(outer);bar.pack(fill="x")
        self.choose=ttk.Button(bar,text="Choose photo and measure",command=self.pick)
        self.choose.pack(side="left")
        self.again=ttk.Button(bar,text="Measure again",command=self.run,state="disabled")
        self.again.pack(side="left",padx=8)
        self.open_button=ttk.Button(bar,text="Open saved image",command=self.open_image,state="disabled")
        self.open_button.pack(side="left")
        options=ttk.Frame(outer);options.pack(fill="x",pady=(12,8))
        self.raw=tk.BooleanVar(value=False)
        self.crop=tk.BooleanVar(value=False)
        self.raw_check=ttk.Checkbutton(options,text="Skip lens correction (comparison only)",variable=self.raw)
        self.raw_check.pack(side="left")
        self.crop_check=ttk.Checkbutton(options,text="Try automatic crop (experimental)",variable=self.crop)
        self.crop_check.pack(side="left",padx=20)
        self.status=tk.StringVar(value="Ready. No command typing needed.")
        ttk.Label(outer,textvariable=self.status,font=("Segoe UI",10),wraplength=940).pack(anchor="w",pady=5)
        self.progress=ttk.Progressbar(outer,mode="indeterminate")
        self.progress.pack(fill="x",pady=(0,8))
        self.dimensions=tk.StringVar(value="Long side: --     Short side: --")
        ttk.Label(outer,textvariable=self.dimensions,font=("Segoe UI",19,"bold")).pack(anchor="w")
        self.score=tk.StringVar(value="Detection score: --")
        ttk.Label(outer,textvariable=self.score).pack(anchor="w",pady=(4,8))
        self.canvas=tk.Canvas(outer,bg="#e8ecef",highlightthickness=0,height=460)
        self.canvas.pack(fill="both",expand=True)
        self.canvas.create_text(430,200,text="Your photo result will appear here",fill="#56616b",
                                font=("Segoe UI",14))
        ttk.Label(outer,text="Detection confidence is not a size-accuracy guarantee. Check the result against your ruler.",
                  wraplength=940).pack(anchor="w",pady=(10,0))
        self.saved=tk.StringVar(value="Results will be saved beside this launcher.")
        ttk.Label(outer,textvariable=self.saved,wraplength=940).pack(anchor="w",pady=(4,0))
        root.after(100,self.poll)

    def pick(self):
        path=filedialog.askopenfilename(title="Choose an original envelope photo",
            filetypes=[("Camera photos","*.HEIC *.heic *.JPG *.jpg *.JPEG *.jpeg *.PNG *.png"),("All files","*.*")])
        if path:
            self.photo=Path(path)
            self.run()

    def busy(self,yes):
        state="disabled" if yes else "normal"
        for widget in (self.choose,self.again,self.raw_check,self.crop_check):
            widget.configure(state=state)
        self.open_button.configure(state="disabled" if yes or self.last_overlay is None else "normal")
        if yes:self.progress.start(15)
        else:self.progress.stop()

    def run(self):
        if not self.photo:
            return
        self.busy(True)
        self.dimensions.set("Measuring...")
        self.score.set("Detection score: --")
        self.status.set(f"Processing {self.photo.name}. The first photo also loads the model; please wait.")
        threading.Thread(target=self.worker,args=(self.photo,self.raw.get(),self.crop.get()),daemon=True).start()

    def worker(self,photo,raw,crop):
        try:
            import cv2
            from calibration.camera import read_image
            from inference.measure import Meter
            if self.meter is None:
                self.meter=Meter(self.repo/"calibration/parameters.json",
                                 self.repo/"models/runs/first/best.pth",36.,36.,"cpu")
            result,overlay=self.meter.measure(read_image(photo),raw=raw,refine_crop=crop)
            text=result["status"] if result["status"]!="ok" else (
                f"{result['width_mm']:.1f} x {result['height_mm']:.1f} mm  score {result['confidence']:.2f}")
            cv2.putText(overlay,text,(15,32),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,0),4,cv2.LINE_AA)
            cv2.putText(overlay,text,(15,32),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),1,cv2.LINE_AA)
            self.output.mkdir(parents=True,exist_ok=True)
            suffix=("-raw" if raw else "-corrected")+("-crop" if crop else "")
            destination=self.output/(photo.stem+suffix+".jpg")
            if not cv2.imwrite(str(destination),overlay):
                raise IOError("Could not save the result image.")
            (destination.with_suffix(".json")).write_text(
                json.dumps(dict(source=photo.name,**result),indent=2,allow_nan=False)+"\n",encoding="utf-8")
            self.events.put(("result",result,destination,overlay[:,:,::-1].copy()))
        except Exception as error:
            self.output.mkdir(parents=True,exist_ok=True)
            log=self.output/"last-error.txt"
            log.write_text(traceback.format_exc(),encoding="utf-8")
            self.events.put(("error",str(error),log))

    def poll(self):
        try:
            event=self.events.get_nowait()
        except queue.Empty:
            pass
        else:
            self.busy(False)
            if event[0]=="error":
                self.dimensions.set("No measurement")
                self.status.set("Could not process this photo.")
                messagebox.showerror("Measure It",f"{event[1]}\n\nDetails saved in {event[2]}")
            else:
                _,result,destination,rgb=event
                from PIL import Image,ImageTk
                self.last_overlay=destination
                self.open_button.configure(state="normal")
                if result["status"]=="ok":
                    self.dimensions.set(f"Long side: {result['width_mm']:.1f} mm     Short side: {result['height_mm']:.1f} mm")
                else:
                    self.dimensions.set("Measurement refused")
                score=result["confidence"]
                self.score.set("Detection score: "+("--" if score is None else f"{score:.3f}"))
                self.status.set(f"{self.photo.name} | {result['correction']} | {result['status']}")
                self.saved.set("Saved image and JSON: "+str(destination.parent))
                picture=Image.fromarray(rgb)
                picture.thumbnail((max(700,self.canvas.winfo_width()-10),max(400,self.canvas.winfo_height()-10)))
                self.image_ref=ImageTk.PhotoImage(picture)
                self.canvas.delete("all")
                self.canvas.create_image(self.canvas.winfo_width()//2,self.canvas.winfo_height()//2,image=self.image_ref)
        self.root.after(100,self.poll)

    def open_image(self):
        import os
        if self.last_overlay:
            os.startfile(str(self.last_overlay))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photo",type=Path)
    parser.add_argument("--out",type=Path,default=Path("demo-results"))
    args=parser.parse_args()
    root=tk.Tk()
    app=Desktop(root,args.out)
    if args.photo:
        app.photo=args.photo.resolve()
        root.after(200,app.run)
    root.mainloop()


if __name__=="__main__":
    main()

