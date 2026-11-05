"""Fit lens correction on the twenty full-frame training views of the printed board."""
import argparse,csv,json
from pathlib import Path
import cv2
import numpy as np
from calibration.camera import find_board,read_image,correct

def fit_board(originals, output, evidence, source_map=Path("dataset/capture-index.csv"), square_mm=18.0):
    from dataset.iphone import sensor_frame
    rows=[row for row in csv.DictReader(source_map.open()) if row["board"]=="1" and row["split"]=="train"]
    obj=np.zeros((63,3),np.float32)
    obj[:,:2]=np.mgrid[0:7,0:9].T.reshape(-1,2)*square_mm
    views=[]; names=[]; first=None
    for row in rows:
        p=originals/row["source"]
        if (originals/(p.stem+".png")).exists(): frame=read_image(originals/(p.stem+".png"))
        else: frame=sensor_frame(p)
        if first is None:first=frame
        if frame.shape[:2]!=(900,1200): raise ValueError("Expected normalized 1200 x 900 sensor coordinates")
        pts=find_board(frame)
        if pts is not None:
            views.append(pts);names.append(p.name)
    if len(views)<20: raise ValueError(f"Only {len(views)} complete checkerboards")
    # Equivalent focal length for the 35 mm format is rounded by the phone.
    focal=26*np.hypot(1200,900)/np.hypot(36,24)
    K0=np.array([[focal,0,599.5],[0,focal,449.5],[0,0,1]],np.float64)
    flags=(cv2.CALIB_USE_INTRINSIC_GUESS|cv2.CALIB_FIX_FOCAL_LENGTH|cv2.CALIB_FIX_ASPECT_RATIO|
           cv2.CALIB_FIX_PRINCIPAL_POINT|cv2.CALIB_ZERO_TANGENT_DIST|cv2.CALIB_FIX_K2|cv2.CALIB_FIX_K3)
    def fit(indices,constrained=True):
        return cv2.calibrateCamera([obj]*len(indices),[views[i] for i in indices],(1200,900),
                                   K0.copy() if constrained else None,None,flags=flags if constrained else 0)
    idx=list(range(len(views)))
    rms,K,D,rot,trans=fit(idx)
    rms_free,K_free,D_free,_,_=fit(idx,False)
    per={}
    for i in idx:
        projected=cv2.projectPoints(obj,rot[i],trans[i],K,D)[0]
        per[names[i]]=float(np.sqrt(np.mean(np.sum((projected-views[i])**2,axis=2))))
    par=np.mgrid[0:7,0:9].T.reshape(-1,2).sum(axis=1)%2==0
    held=[]
    for i in idx:
        _,Ki,Di,_,_=fit([k for k in idx if k!=i])
        scores={}
        for key,dist in [("corrected",Di),("zero_distortion",np.zeros_like(Di))]:
            errors=[]
            for used in (par,~par):
                ok,v,t=cv2.solvePnP(obj[used],views[i][used],Ki,dist)
                if not ok:raise ValueError("Pose fit failed")
                q=cv2.projectPoints(obj[~used],v,t,Ki,dist)[0]
                errors.extend(np.sum((q-views[i][~used])**2,axis=2).ravel().tolist())
            scores[key]=float(np.sqrt(np.mean(errors)))
        held.append(dict(file=names[i],**scores))
    spread=np.array(views).reshape(-1,2)
    basis="nominal print specification; physical reading unavailable"
    reading_path=Path("measurement/physical-readings.json")
    if reading_path.exists():
        board=json.loads(reading_path.read_text(encoding="utf-8"))["checkerboard"]
        if all(np.isclose(board[k],square_mm) for k in ("square_width_mm","square_height_mm")):
            basis=(f"User ruler reading: {board['horizontal_span_mm']:g} mm over "
                   f"{board['squares_per_span']} consecutive squares in each axis = {square_mm:g} mm; "
                   "measurement/physical-readings.json. No calliper used.")
    params=dict(image_size=[1200,900],native_image_size=[4032,3024],square_mm=square_mm,
                square_size_basis=basis,
                fit="single radial coefficient; fixed centre, zero tangent, square pixels and 26 mm-equivalent EXIF focal prior",
                camera_matrix=K.tolist(),distortion=D.ravel().tolist(),rms_px=float(rms),accepted=names,rejected=[row['source'] for row in rows if row['source'] not in names],per_view_rms_px=per)
    output.parent.mkdir(parents=True,exist_ok=True);evidence.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(params,indent=2,allow_nan=False))
    details=dict(unconstrained_rms_px=float(rms_free),unconstrained_matrix=K_free.tolist(),
                 unconstrained_distortion=D_free.ravel().tolist(),corner_extent_px=[spread.min(0).tolist(),spread.max(0).tolist()],
                 leave_one_view_out=held)
    (evidence/"fullframe-validation.json").write_text(json.dumps(details,indent=2,allow_nan=False))
    pair=cv2.hconcat([first,correct(first,params)])
    cv2.imwrite(str(output.with_suffix(".preview.jpg")),cv2.resize(pair,(1200,450),interpolation=cv2.INTER_AREA))
    print(json.dumps(dict(views=len(views),rms=rms,K=K.tolist(),D=D.ravel().tolist(),
                          free_rms=rms_free,freeK=K_free.tolist(),freeD=D_free.ravel().tolist(),
                          heldout_corrected=np.mean([a['corrected'] for a in held]),heldout_raw=np.mean([a['zero_distortion'] for a in held]))))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("originals",type=Path)
    p.add_argument("--output",type=Path,default=Path("calibration/parameters.json"))
    p.add_argument("--evidence",type=Path,default=Path("calibration/evidence"))
    p.add_argument("--square-mm",type=float,default=18.)
    a=p.parse_args()
    fit_board(a.originals,a.output,a.evidence,square_mm=a.square_mm)

