"""Evaluate recorded image-frame estimates without keyframe interpolation."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from splg_slam.geometry.frame_evaluation import (
    camera_centers, error_stats, euroc_camera_gt, paired_frame_indices,
    rigid_alignment, valid_estimates,
)


def evaluate(directory, compare=None, output=None):
    directory=Path(directory);output=Path(output or directory);output.mkdir(parents=True,exist_ok=True)
    data=dict(np.load(directory/"trajectory_frames.npz",allow_pickle=False))
    cfg=yaml.safe_load((directory/"config_effective.yaml").read_text())
    valid=valid_estimates(data)
    summary=dict(input_frames=len(valid), visual_final_valid=int(valid.sum()),
                 status_counts=dict(Counter(data["status"].tolist())),
                 protocol="Final reference-corrected image poses, exact image timestamps, SE(3) alignment (scale=1), no estimate interpolation, no trimming. Report coverage with errors.",
                 online_warning="Online poses are raw frontend poses in a changing gauge; not evaluated with one global alignment.",
                 final_warning="Non-keyframes retain their measured pose relative to a reference keyframe; they are not independently optimized.")
    ts=data["timestamp_ns"];time=(ts-ts[0])/1e9;xyz=camera_centers(data["pose_final_cw"])
    fig,axes=plt.subplots(2,1,figsize=(13,7),layout="constrained",sharex=True)
    for axis,label in enumerate("XYZ"):
        axes[0].plot(time,xyz[:,axis],lw=.8,label=label)
    axes[0].set(ylabel="Position (m)",title="Final reference-corrected camera positions");axes[0].legend()
    names=["initialized","tracked","relocalized","lost","new_segment","imu_only","bootstrap_wait"]
    codes=np.array([names.index(s) for s in data["status"]])
    axes[1].scatter(time,codes,s=5);axes[1].set(yticks=np.arange(len(names)),yticklabels=names,xlabel="Time (s)",title="One status per input image; lost frames remain missing")
    for ax in axes:ax.grid(alpha=.25)
    fig.savefig(output/"tracking_frames.png",dpi=145);plt.close(fig)
    ds=cfg['dataset'];gt_file=Path(ds.get('mav0_dir',''))/'state_groundtruth_estimate0/data.csv'
    if ds.get('kind','euroc')=='euroc' and gt_file.exists():
        gt,gt_valid=euroc_camera_gt(ds['mav0_dir'],ts)
        mask=valid & gt_valid
        summary.update(gt_available=True,gt_eligible_frames=int(gt_valid.sum()),matched_frames=int(mask.sum()),
                       gt_coverage=float(mask.sum()/max(gt_valid.sum(),1)))
        if mask.sum() >= 3:
            aligned,rotation,translation=rigid_alignment(xyz[mask],gt[mask])
            error=np.linalg.norm(aligned-gt[mask],axis=1)
            summary['ate_se3']=error_stats(error)
            full_error=np.full(len(ts),np.nan);full_error[mask]=error
            np.savez_compressed(output/'frame_gt_comparison.npz',timestamp_ns=ts,valid=mask,ground_truth=gt,error_m=full_error,rotation=rotation,translation=translation)
            series=[(directory.name,time[mask],error,aligned,gt[mask])]
            if compare is not None:
                ref=dict(np.load(Path(compare)/'trajectory_frames.npz',allow_pickle=False))
                common,ia,ib=paired_frame_indices(data,ref)
                cg,cvalid=euroc_camera_gt(ds['mav0_dir'],common)
                ia,ib,common,cg=ia[cvalid],ib[cvalid],common[cvalid],cg[cvalid]
                summary['common_frames']=len(common)
                summary['common_gt_coverage']=len(common)/max(int(gt_valid.sum()),1)
                if len(common)>=3:
                    cx=camera_centers(data['pose_final_cw'][ia]);rx=camera_centers(ref['pose_final_cw'][ib])
                    ca,_,_=rigid_alignment(cx,cg);ra,_,_=rigid_alignment(rx,cg)
                    ce=np.linalg.norm(ca-cg,axis=1);re=np.linalg.norm(ra-cg,axis=1)
                    summary['common_current']=error_stats(ce);summary['common_reference']=error_stats(re)
                    summary['reference_directory']=str(compare)
                    np.savez_compressed(output/'paired_frame_gt.npz',timestamp_ns=common,current_frame_ids=data['frame_id'][ia],reference_frame_ids=ref['frame_id'][ib],ground_truth=cg,current_aligned=ca,reference_aligned=ra,current_error_m=ce,reference_error_m=re)
                    ct=(common-ts[0])/1e9
                    series=[(f'{Path(compare).name} (same frames)',ct,re,ra,cg),(f'{directory.name} (same frames)',ct,ce,ca,cg)]
            fig,(a,b)=plt.subplots(1,2,figsize=(14,5),layout='constrained')
            for label,t,e,pos,g in series:
                a.plot(t,e,lw=.8,label=label);b.plot(pos[:,0],pos[:,1],lw=.7,label=label)
            b.plot(g[:,0],g[:,1],'k--',lw=.7,label='Camera GT');b.set_aspect('equal',adjustable='datalim')
            a.set(xlabel='Time (s)',ylabel='Position error (m)',title='SE(3) ATE at actual image timestamps')
            b.set(xlabel='X (m)',ylabel='Y (m)',title='Trajectory vs GT; scale fixed')
            for ax in (a,b):ax.grid(alpha=.25);ax.legend(fontsize=8)
            fig.savefig(output/'frames_vs_gt.png',dpi=145);plt.close(fig)
    else:summary['gt_available']=False
    (output/'frame_metrics.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path,help='build output containing trajectory_frames.npz and config_effective.yaml')
    p.add_argument('--compare',type=Path,help='reference run of the same sequence; compare exact common valid image timestamps')
    p.add_argument('--output-dir',type=Path)
    a=p.parse_args();print(json.dumps(evaluate(a.directory,a.compare,a.output_dir),indent=2))
