from collections import defaultdict
import numpy as np
from copy import deepcopy
import networkx as nx
import matplotlib.pyplot as plt
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector, quaternion_from_coeff
import torch
from .utils import generate_bev_path

MAX_DIST = 30
MAX_STEP = 10


def calc_position_distance(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    return dist


def calculate_vp_rel_pos_fts(
    a, b, base_heading=0, base_elevation=0, to_clock=False, return_xz_dist=False
):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    xz_dist = max(np.sqrt(dx**2 + dz**2), 1e-08)
    xyz_dist = max(np.sqrt(dx**2 + dy**2 + dz**2), 1e-08)
    heading = np.arcsin(-dx / xz_dist)
    if b[2] > a[2]:
        heading = np.pi - heading
    heading -= base_heading
    if to_clock:
        heading = 2 * np.pi - heading
    elevation = np.arcsin(dz / xyz_dist)
    elevation -= base_elevation
    if return_xz_dist:
        return (heading, elevation, xz_dist)
    else:
        return (heading, elevation, xyz_dist)


def get_angle_fts(headings, elevations, angle_feat_size):
    ang_fts = [np.sin(headings), np.cos(headings), np.sin(elevations), np.cos(elevations)]
    ang_fts = np.vstack(ang_fts).transpose().astype(np.float32)
    num_repeats = angle_feat_size // 4
    if num_repeats > 1:
        ang_fts = np.concatenate([ang_fts] * num_repeats, 1)
    return ang_fts


def heading_from_quaternion(quat: np.array):
    quat = quaternion_from_coeff(quat)
    heading_vector = quaternion_rotate_vector(quat.inverse(), np.array([0, 0, -1]))
    phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
    return phi % (2 * np.pi)


def estimate_cand_pos(pos, ori, ang, dis):
    cand_num = len(ang)
    cand_pos = np.zeros([cand_num, 3])
    ang = np.array(ang)
    dis = np.array(dis)
    ang = (heading_from_quaternion(ori) + ang) % (2 * np.pi)
    cand_pos[:, 0] = pos[0] - dis * np.sin(ang)
    cand_pos[:, 1] = pos[1]
    cand_pos[:, 2] = pos[2] - dis * np.cos(ang)
    return cand_pos


class FloydGraph(object):

    def __init__(self):
        self._dis = defaultdict(lambda: defaultdict(lambda: 95959595))
        self._point = defaultdict(lambda: defaultdict(lambda: ""))
        self._visited = set()

    def distance(self, x, y):
        if x == y:
            return 0
        else:
            return self._dis[x][y]

    def add_edge(self, x, y, dis):
        if dis < self._dis[x][y]:
            self._dis[x][y] = dis
            self._dis[y][x] = dis
            self._point[x][y] = ""
            self._point[y][x] = ""

    def update(self, k):
        for x in self._dis:
            for y in self._dis:
                if x != y and x != k and (y != k):
                    t_dis = self._dis[x][y] + self._dis[y][k]
                    if t_dis < self._dis[x][k]:
                        self._dis[x][k] = t_dis
                        self._dis[k][x] = t_dis
                        self._point[x][k] = y
                        self._point[k][x] = y
        for x in self._dis:
            for y in self._dis:
                if x != y:
                    t_dis = self._dis[x][k] + self._dis[k][y]
                    if t_dis < self._dis[x][y]:
                        self._dis[x][y] = t_dis
                        self._dis[y][x] = t_dis
                        self._point[x][y] = k
                        self._point[y][x] = k
        self._visited.add(k)

    def visited(self, k):
        return k in self._visited

    def path(self, x, y):
        if x == y:
            return []
        if self._point[x][y] == "":
            return [y]
        else:
            k = self._point[x][y]
            return self.path(x, k) + self.path(k, y)


class GraphMap(object):

    def __init__(self, has_real_pos, loc_noise, merge_ghost, ghost_aug):
        self.graph_nx = nx.Graph()
        self.node_pos = {}
        self.node_embeds = {}
        self.node_stepId = {}
        self.ghost_cnt = 0
        self.ghost_pos = {}
        self.ghost_mean_pos = {}
        self.ghost_embeds = {}
        self.ghost_fronts = {}
        self.ghost_trajs = {}
        self.ghost_real_pos = {}
        self.has_real_pos = has_real_pos
        self.merge_ghost = merge_ghost
        self.ghost_aug = ghost_aug
        self.loc_noise = loc_noise
        self.node_pc = {}
        self.shortest_path = None
        self.shortest_dist = None
        self.node_stop_scores = {}

    def _localize(self, qpos, kpos_dict, ignore_height=False):
        min_dis = 10000
        min_vp = None
        for kvp, kpos in kpos_dict.items():
            if ignore_height:
                dis = ((qpos[[0, 2]] - kpos[[0, 2]]) ** 2).sum() ** 0.5
            else:
                dis = ((qpos - kpos) ** 2).sum() ** 0.5
            if dis < min_dis:
                min_dis = dis
                min_vp = kvp
        min_vp = None if min_dis > self.loc_noise else min_vp
        return min_vp

    def identify_node(self, cur_pos, cur_ori, cand_ang, cand_dis):
        cur_vp = str(len(self.node_pos))
        cand_vp = [f"{cur_vp}_{str(i)}" for i in range(len(cand_ang))]
        cand_pos = [p for p in estimate_cand_pos(cur_pos, cur_ori, cand_ang, cand_dis)]
        return (cur_vp, cand_vp, cand_pos)

    def delete_ghost(self, vp):
        self.ghost_pos.pop(vp)
        self.ghost_mean_pos.pop(vp)
        self.ghost_embeds.pop(vp)
        self.ghost_fronts.pop(vp)
        if self.has_real_pos:
            self.ghost_real_pos.pop(vp)

    def update_graph(
        self,
        prev_vp,
        step_id,
        cur_vp,
        cur_pos,
        cur_embeds,
        cand_vp,
        cand_pos,
        cand_embeds,
        cand_real_pos,
        action_trajs,
    ):
        self.graph_nx.add_node(cur_vp)
        if prev_vp is not None:
            prev_pos = self.node_pos[prev_vp]
            dis = calc_position_distance(prev_pos, cur_pos)
            self.graph_nx.add_edge(prev_vp, cur_vp, weight=dis)
        self.node_pos[cur_vp] = cur_pos
        self.node_embeds[cur_vp] = cur_embeds
        self.node_stepId[cur_vp] = step_id
        for i, (cvp, cpos, cembeds) in enumerate(zip(cand_vp, cand_pos, cand_embeds)):
            localized_nvp = self._localize(cpos, self.node_pos)
            if localized_nvp is not None:
                dis = calc_position_distance(cur_pos, self.node_pos[localized_nvp])
                self.graph_nx.add_edge(cur_vp, localized_nvp, weight=dis)
            elif self.merge_ghost:
                localized_gvp = self._localize(cpos, self.ghost_mean_pos)
                if localized_gvp is None:
                    gvp = f"g{str(self.ghost_cnt)}"
                    self.ghost_cnt += 1
                    self.ghost_pos[gvp] = [cpos]
                    self.ghost_mean_pos[gvp] = cpos
                    self.ghost_embeds[gvp] = [cembeds, 1]
                    self.ghost_fronts[gvp] = [cur_vp]
                    self.ghost_trajs[gvp] = {}
                    self.ghost_trajs[gvp][cur_vp] = action_trajs[i]
                    if self.has_real_pos:
                        self.ghost_real_pos[gvp] = [cand_real_pos[i]]
                else:
                    gvp = localized_gvp
                    self.ghost_pos[gvp].append(cpos)
                    self.ghost_mean_pos[gvp] = np.mean(self.ghost_pos[gvp], axis=0)
                    self.ghost_embeds[gvp][0] = self.ghost_embeds[gvp][0] + cembeds
                    self.ghost_embeds[gvp][1] += 1
                    self.ghost_fronts[gvp].append(cur_vp)
                    self.ghost_trajs[gvp][cur_vp] = action_trajs[i]
                    if self.has_real_pos:
                        self.ghost_real_pos[gvp].append(cand_real_pos[i])
            else:
                gvp = f"g{str(self.ghost_cnt)}"
                self.ghost_cnt += 1
                self.ghost_pos[gvp] = [cpos]
                self.ghost_mean_pos[gvp] = cpos
                self.ghost_embeds[gvp] = [cembeds, 1]
                self.ghost_fronts[gvp] = [cur_vp]
                self.ghost_trajs[gvp] = {}
                self.ghost_trajs[gvp][cur_vp] = action_trajs[i]
                if self.has_real_pos:
                    self.ghost_real_pos[gvp] = [cand_real_pos[i]]
        self.ghost_aug_pos = deepcopy(self.ghost_mean_pos)
        if self.ghost_aug != 0:
            for gvp, gpos in self.ghost_aug_pos.items():
                gpos_noise = np.random.normal(
                    loc=(0, 0, 0), scale=(self.ghost_aug, 0, self.ghost_aug), size=(3,)
                )
                gpos_noise[gpos_noise < -self.ghost_aug] = -self.ghost_aug
                gpos_noise[gpos_noise > self.ghost_aug] = self.ghost_aug
                self.ghost_aug_pos[gvp] = gpos + gpos_noise
        self.shortest_path = dict(nx.all_pairs_dijkstra_path(self.graph_nx))
        self.shortest_dist = dict(nx.all_pairs_dijkstra_path_length(self.graph_nx))

    def front_to_ghost_dist(self, ghost_vp):
        min_dis = 10000
        min_front = None
        for front_vp in self.ghost_fronts[ghost_vp]:
            dis = calc_position_distance(self.node_pos[front_vp], self.ghost_aug_pos[ghost_vp])
            if dis < min_dis:
                min_dis = dis
                min_front = front_vp
        return (min_dis, min_front)

    def get_node_embeds(self, vp):
        if not vp.startswith("g"):
            return self.node_embeds[vp]
        else:
            return self.ghost_embeds[vp][0] / self.ghost_embeds[vp][1]

    def get_pos_fts(self, cur_vp, cur_pos, cur_ori, gmap_vp_ids):
        rel_angles, rel_dists = ([], [])
        for vp in gmap_vp_ids:
            if vp is None:
                rel_angles.append([0, 0])
                rel_dists.append([0, 0, 0])
            elif vp.startswith("g"):
                base_heading = heading_from_quaternion(cur_ori)
                base_elevation = 0
                vp_pos = self.ghost_aug_pos[vp]
                rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts(
                    cur_pos, vp_pos, base_heading, base_elevation, to_clock=True
                )
                rel_angles.append([rel_heading, rel_elevation])
                front_dis, front_vp = self.front_to_ghost_dist(vp)
                shortest_dist = self.shortest_dist[cur_vp][front_vp] + front_dis
                shortest_step = len(self.shortest_path[cur_vp][front_vp]) + 1
                rel_dists.append(
                    [rel_dist / MAX_DIST, shortest_dist / MAX_DIST, shortest_step / MAX_STEP]
                )
            else:
                base_heading = heading_from_quaternion(cur_ori)
                base_elevation = 0
                vp_pos = self.node_pos[vp]
                rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts(
                    cur_pos, vp_pos, base_heading, base_elevation, to_clock=True
                )
                rel_angles.append([rel_heading, rel_elevation])
                shortest_dist = self.shortest_dist[cur_vp][vp]
                shortest_step = len(self.shortest_path[cur_vp][vp])
                rel_dists.append(
                    [rel_dist / MAX_DIST, shortest_dist / MAX_DIST, shortest_step / MAX_STEP]
                )
        rel_angles = np.array(rel_angles).astype(np.float32)
        rel_dists = np.array(rel_dists).astype(np.float32)
        rel_ang_fts = get_angle_fts(rel_angles[:, 0], rel_angles[:, 1], angle_feat_size=4)
        return np.concatenate([rel_ang_fts, rel_dists], 1)

    def get_traj_fts(self, cur_vp, cur_pos, cur_ori, gmap_vp_ids):
        rel_angles, rel_dists = ([], [])
        wp_traj_fts, wp_traj_len = ([], [])
        for vp in gmap_vp_ids:
            if vp is None:
                continue
            elif vp.startswith("g"):
                base_heading = heading_from_quaternion(cur_ori)
                base_elevation = 0
                vp_pos = self.ghost_aug_pos[vp]
                front_dis, front_vp = self.front_to_ghost_dist(vp)
                rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts(
                    cur_pos, self.node_pos[front_vp], base_heading, base_elevation, to_clock=True
                )
                shortest_dist = self.shortest_dist[cur_vp][front_vp]
                shortest_step = len(self.shortest_path[cur_vp][front_vp]) + 1
                if rel_dist == 1e-08:
                    rel_angles.append([0, 0])
                    rel_dists.append([0, 0, 0])
                else:
                    rel_angles.append([rel_heading, rel_elevation])
                    rel_dists.append(
                        [rel_dist / MAX_DIST, shortest_dist / MAX_DIST, shortest_step / MAX_STEP]
                    )
                traj_fts = self.get_traj_tokens_ref(front_vp, vp, cur_vp, cur_pos, cur_ori)
                wp_traj_fts.append(traj_fts)
                wp_traj_len.append(traj_fts.shape[0])
            else:
                base_heading = heading_from_quaternion(cur_ori)
                base_elevation = 0
                vp_pos = self.node_pos[vp]
                rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts(
                    cur_pos, vp_pos, base_heading, base_elevation, to_clock=True
                )
                rel_angles.append([rel_heading, rel_elevation])
                shortest_dist = self.shortest_dist[cur_vp][vp]
                shortest_step = len(self.shortest_path[cur_vp][vp])
                rel_dists.append(
                    [rel_dist / MAX_DIST, shortest_dist / MAX_DIST, shortest_step / MAX_STEP]
                )
                traj_fts = np.zeros([1, 7])
                wp_traj_fts.append(traj_fts)
                wp_traj_len.append(0)
        rel_angles = np.array(rel_angles).astype(np.float32)
        rel_dists = np.array(rel_dists).astype(np.float32)
        rel_ang_fts = get_angle_fts(rel_angles[:, 0], rel_angles[:, 1], angle_feat_size=4)
        front_vp_fts = np.concatenate([rel_ang_fts, rel_dists], 1)
        return (front_vp_fts, wp_traj_fts, wp_traj_len)

    def get_traj_tokens(
        self, front_vp, vp, eps=1e-08, base_heading=0, base_elevation=0, to_clock=False
    ):
        ghost_trajs = self.ghost_trajs[vp][front_vp]
        start_pos = self.node_pos[front_vp]
        points = np.concatenate([start_pos.reshape(1, 3), ghost_trajs], axis=0)
        deltas = points[1:] - points[:-1]
        feats = []
        for delta in deltas:
            dx, dy, dz = (delta[0], delta[1], delta[2])
            xz_dist = max(np.sqrt(dx**2 + dz**2), eps)
            xyz_dist = max(np.sqrt(dx**2 + dy**2 + dz**2), eps)
            heading = np.arcsin(-dx / xz_dist)
            if dz > 0:
                heading = np.pi - heading
            heading -= base_heading
            if to_clock:
                heading = 2 * np.pi - heading
            elevation = np.arcsin(dz / xyz_dist)
            elevation -= base_elevation
            sin_h, cos_h = (np.sin(heading), np.cos(heading))
            sin_e, cos_e = (np.sin(elevation), np.cos(elevation))
            fts = np.array([dx, dy, dz, xyz_dist, sin_h, cos_h, sin_e, cos_e])
            feats.append(fts)
        return np.stack(feats, axis=0)

    def get_traj_tokens_ref(
        self,
        front_vp,
        vp,
        cur_vp,
        cur_pos,
        cur_ori,
        eps=1e-08,
        base_heading=0,
        base_elevation=0,
        to_clock=False,
    ):
        ghost_trajs = self.ghost_trajs[vp][front_vp]
        for wp in ghost_trajs:
            rel_angles, rel_dists = ([], [])
            base_heading = heading_from_quaternion(cur_ori)
            base_elevation = 0
            rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts(
                cur_pos, wp, base_heading, base_elevation, to_clock=True
            )
            rel_angles.append([rel_heading, rel_elevation])
            front_dis, front_vp = self.front_to_ghost_dist(vp)
            shortest_dist = self.shortest_dist[cur_vp][front_vp] + front_dis
            shortest_step = len(self.shortest_path[cur_vp][front_vp]) + 1
            rel_dists.append(
                [rel_dist / MAX_DIST, shortest_dist / MAX_DIST, shortest_step / MAX_STEP]
            )
        rel_angles = np.array(rel_angles).astype(np.float32)
        rel_dists = np.array(rel_dists).astype(np.float32)
        rel_ang_fts = get_angle_fts(rel_angles[:, 0], rel_angles[:, 1], angle_feat_size=4)
        return np.concatenate([rel_ang_fts, rel_dists], 1)

    def update_node_pc(self, vp, pc, pc_mask, pc_feat):
        self.node_pc[vp] = [pc, pc_mask, pc_feat]

    def gather_node_pc(self, vp, order):
        if order == 0:
            return self.node_pc[vp]
        else:
            cvps = [cvp for cvp in self.node_pc.keys() if len(self.shortest_path[vp][cvp]) <= order]
            pc = [self.node_pc[cvp][0] for cvp in cvps]
            pc_mask = [self.node_pc[cvp][1] for cvp in cvps]
            pc_feat = [self.node_pc[cvp][2] for cvp in cvps]
            pc = torch.cat(pc, dim=0)
            pc_mask = torch.cat(pc_mask, dim=0)
            pc_feat = torch.cat(pc_feat, dim=0)
            return (pc, pc_mask, pc_feat)

    def get_neighbors(self, cur_vp, cur_pos, cur_ori):
        cands_vp = [None]
        cands_relpos = [np.zeros(2)]
        base_heading = heading_from_quaternion(cur_ori)
        base_elevation = 0
        for vp, vp_pos in self.node_pos.items():
            if len(self.shortest_path[cur_vp][vp]) == 2:
                cands_vp.append(vp)
                rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts(
                    cur_pos,
                    vp_pos,
                    base_heading,
                    base_elevation,
                    to_clock=True,
                    return_xz_dist=True,
                )
                cands_relpos.append(np.array([rel_heading, rel_dist], dtype=np.float32))
        for vp, vp_pos in self.ghost_aug_pos.items():
            if cur_vp in self.ghost_fronts[vp]:
                cands_vp.append(vp)
                rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts(
                    cur_pos,
                    vp_pos,
                    base_heading,
                    base_elevation,
                    to_clock=True,
                    return_xz_dist=True,
                )
                cands_relpos.append(np.array([rel_heading, rel_dist], dtype=np.float32))
        return (cands_vp, np.array(cands_relpos))

    def get_neighbor_trajs(self, cur_vp, cands_vp):
        cand_trajs = []
        for vp in cands_vp:
            if vp is None:
                continue
            if vp.startswith("g"):
                cand_traj = self.ghost_trajs[vp][cur_vp]
            else:
                shortest_path = self.shortest_path[cur_vp][vp]
                cur_vp, target_vp = (shortest_path[0], shortest_path[1])
                path = generate_bev_path(self.node_pos[cur_vp], self.node_pos[target_vp])
                cand_traj = np.concatenate([self.node_pos[cur_vp].reshape(1, 3), path], axis=0)
            cand_trajs.append(cand_traj)
        return cand_trajs
