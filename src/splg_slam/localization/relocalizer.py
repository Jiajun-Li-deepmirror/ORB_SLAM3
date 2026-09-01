import numpy as np
import torch

from splg_slam.features.splg import SPLG
from splg_slam.geometry.pnp import solve_pnp_ransac
from splg_slam.geometry.stereo import StereoRectifier
from splg_slam.localization.retrieval import GlobalDescriptorExtractor, GlobalDescriptorIndex
from splg_slam.map.world_map import WorldMap


class Relocalizer:
    """Localizes a single query image (already rectified, left-camera frame) against a
    pre-built map: global-descriptor retrieval -> LightGlue match against each candidate
    keyframe's stored features -> 2D-3D correspondences via the candidate's map points
    -> PnP+RANSAC. The candidate with the most PnP inliers wins."""

    def __init__(self, world_map: WorldMap, rectifier: StereoRectifier, cfg):
        self.world_map = world_map
        self.rectifier = rectifier
        self.cfg = cfg
        self.splg = SPLG(max_keypoints=cfg.features.max_keypoints)
        self.global_extractor = GlobalDescriptorExtractor()
        self.index = GlobalDescriptorIndex()
        self.index.build(world_map)
        self._kf_feats_cache: dict[int, dict] = {}
        # LightGlue only uses image_size to normalize keypoint coordinates; every keyframe
        # (and every query) is rectified to this same fixed size, so it's computed once here
        # rather than stored per-keyframe.
        h, w = self.rectifier.map_l[0].shape
        self._image_size = torch.tensor([[float(w), float(h)]], device=self.splg.device)

    def _feats_for_keyframe(self, kf_id: int) -> dict:
        """Builds a LightGlue-ready feature dict straight from the keyframe's stored
        keypoints/descriptors (captured once at map-build time), instead of reloading the
        image and re-running SuperPoint on it - the map already has everything LightGlue's
        matcher needs (keypoints, descriptors, image_size)."""
        if kf_id not in self._kf_feats_cache:
            kf = self.world_map.keyframes[kf_id]
            self._kf_feats_cache[kf_id] = {
                "keypoints": torch.from_numpy(kf.keypoints).float()[None].to(self.splg.device),
                "descriptors": torch.from_numpy(kf.descriptors).float()[None].to(self.splg.device),
                "image_size": self._image_size,
            }
        return self._kf_feats_cache[kf_id]

    def localize(self, rect_img_left: np.ndarray, top_k: int = 5) -> tuple[bool, np.ndarray | None, dict]:
        query_feats = self.splg.extract(rect_img_left)
        kpts_q, _ = SPLG.to_frame_arrays(query_feats)
        query_global = self.global_extractor.extract(rect_img_left)

        candidates = self.index.query(query_global, top_k=top_k)
        if not candidates:
            return False, None, {"reason": "no candidates in retrieval index"}

        best = None
        for kf_id, sim in candidates:
            kf_feats = self._feats_for_keyframe(kf_id)
            matches = self.splg.match(kf_feats, query_feats)["matches"]

            kf = self.world_map.keyframes[kf_id]
            obj_pts, img_pts = [], []
            for kf_i, q_i in matches:
                mp_id = kf.map_point_ids[kf_i]
                if mp_id >= 0:
                    obj_pts.append(self.world_map.map_points[mp_id].position)
                    img_pts.append(kpts_q[q_i])

            if len(obj_pts) < self.cfg.tracking.min_inlier_matches:
                continue

            ok, pose_cw, inlier_mask = solve_pnp_ransac(
                np.asarray(obj_pts), np.asarray(img_pts), self.rectifier.K_rect,
                reproj_threshold_px=self.cfg.tracking.pnp_reproj_threshold_px,
            )
            if not ok:
                continue

            num_inliers = int(inlier_mask.sum())
            if best is None or num_inliers > best["num_inliers"]:
                best = {"pose_cw": pose_cw, "num_inliers": num_inliers, "candidate_kf": kf_id, "retrieval_sim": sim}

            # Candidates are tried in descending retrieval-similarity order, so once one
            # clears a comfortable inlier margin there's no accuracy reason to pay for
            # LightGlue matching against the rest - a landslide PnP consensus this strong
            # only happens against the true location. Profiling on MH04 showed the top
            # retrieval candidate alone landing >=171 inliers in every one of 170 samples,
            # well above this threshold, while a genuinely weak/wrong top candidate still
            # falls through to the full top_k sweep exactly as before.
            if num_inliers >= self.cfg.tracking.relocalization_early_exit_inliers:
                break

        if best is None or best["num_inliers"] < self.cfg.tracking.min_inlier_matches:
            return False, None, {"reason": "no candidate produced enough PnP inliers", "candidates": candidates}

        return True, best["pose_cw"], best
