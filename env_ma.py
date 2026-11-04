"""
================================================================================
Human-Robot Collaborative Disassembly Line Balancing Problem (HRC-DLBP)
Environment: Discrete-Event Simulation Engine with Dynamic Action Masking
Description: Implements topological precedence, workstation capacity, spatial
             operator transit delay (t_m >= 1), and collision avoidance.
================================================================================
"""

import numpy as np
from dataset import parse_opl_data


class DLBP_FinalEnv:
    """
    Simulation Environment for HRC-DLBP under S-Shaped Layouts.

    State Representation:
        - Completed tasks mask: [I]
        - Discarded tasks mask: [I]
        - Operator available time normalized: [O]
        - Operator spatial workstation location: [O]
    Total State Dimension: 2 * I + 2 * O
    Total Action Dimension: I + 1 (0 to I-1: task indices; I: Global Stop Action)
    """

    def __init__(self, case_id='data1'):
        """
        Load OPL instance parameters and precompute static graph attributes.
        """
        data = parse_opl_data(case_id)

        # Scale scalars
        self.P = int(data.get("P", 1))  # Number of product types
        self.I = int(data["I"])  # Total disassembly tasks
        self.J = int(data["J"])  # Recoverable components
        self.O = int(data["O"])  # Total operators (Robots + Mobile humans)
        self.K = int(data["K"])  # Workstations
        self.T = int(data["T"])  # Planning horizon
        self.L = int(data.get("L", 1))  # Disassembly lines

        # Precedence and capability sets
        self.Ip = data.get("Ip", [list(range(self.I))])
        self.Ii_P = data["Ii_P"]  # Precedence relations: AND/OR graph
        self.Ii_C = data["Ii_C"]  # Conflict sets
        self.Kl = data.get("Kl", [list(range(self.K))])
        self.Ko = data["Ko"]  # Workstation accessibility per operator
        self.KO = data.get("KO", [[] for _ in range(self.O)])
        self.Kk = data.get("Kk", [[] for _ in range(self.K)])

        # Physical matrices (Costs, Durations, Delays, Capacities, Salvage)
        self.v = np.asarray(data["v"], dtype=np.float32)  # Component salvage revenue
        self.c = np.asarray(data["c"], dtype=np.float32)  # Operation cost: [O, K, I]
        self.t_d = np.asarray(data["t_d"], dtype=np.int32)  # Processing time: [K, I]
        self.t_m = np.asarray(data["t_m"], dtype=np.int32)  # Transit delay: [K, K]
        self.ck_max = np.asarray(data["ck_max"], dtype=np.int32)  # Station max worker capacity
        self.ci_max = np.asarray(data.get("ci_max", [self.O] * self.I), dtype=np.int32)
        self.Dj = np.asarray(data["Dj"], dtype=np.float32)  # Component-task relation: [J, I]

        # Precomputed profits and topological ordering
        self.task_profit = self._precompute_profits()
        self.task_to_product = self._build_task_to_product()
        self.station_to_lines = self._build_station_to_lines()
        self.base_topological_order = self._build_topological_order()

        # Interface dimensions
        self.action_dim = self.I + 1
        self.state_dim = self.I * 2 + self.O * 2

    def _build_task_to_product(self):
        mapping = np.full(self.I, -1, dtype=np.int32)
        for p, tasks in enumerate(self.Ip):
            for task in tasks:
                if 0 <= task < self.I:
                    mapping[task] = p
        return mapping

    def _build_station_to_lines(self):
        station_to_lines = [set() for _ in range(self.K)]
        for l, stations in enumerate(self.Kl):
            for k in stations:
                if 0 <= k < self.K:
                    station_to_lines[k].add(l)
        return [sorted(list(lines)) for lines in station_to_lines]

    def _precompute_profits(self):
        profits = np.zeros(self.I, dtype=np.float32)
        for i in range(self.I):
            profits[i] = float(np.sum(self.v * self.Dj[:, i]))
        return profits

    def _build_topological_order(self):
        indeg = np.zeros(self.I, dtype=np.int32)
        succ = [[] for _ in range(self.I)]
        for i in range(self.I):
            for pre in self.Ii_P[i]:
                if 0 <= pre < self.I:
                    indeg[i] += 1
                    succ[pre].append(i)

        queue = [i for i in range(self.I) if indeg[i] == 0]
        queue.sort()
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for nxt in succ[node]:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    queue.append(nxt)
            queue.sort()

        if len(order) < self.I:
            seen = set(order)
            order.extend([i for i in range(self.I) if i not in seen])
        return order

    def _refresh_order_matrix(self):
        """Update full ordering matrix a[i1][i2] matching MILP precedence constraints."""
        done_set = {i for i in range(self.I) if self.completed[i]}
        done_order = sorted(done_set, key=lambda i: (int(self.task_end_time[i]), i))
        remain_order = [i for i in self.base_topological_order if i not in done_set]
        order = done_order + remain_order

        pos = np.zeros(self.I, dtype=np.int32)
        for idx, task in enumerate(order):
            pos[task] = idx

        self.a.fill(False)
        for i1 in range(self.I):
            for i2 in range(self.I):
                if i1 != i2:
                    self.a[i1, i2] = pos[i1] < pos[i2]

    def reset(self):
        """Reset environment to initial state at time t=0."""
        self.completed = np.zeros(self.I, dtype=bool)
        self.discarded = np.zeros(self.I, dtype=bool)

        self.task_start_time = np.full(self.I, -1, dtype=np.int32)
        self.task_end_time = np.zeros(self.I, dtype=np.int32)

        self.op_times = np.zeros(self.O, dtype=np.int32)
        self.op_locs = np.zeros(self.O, dtype=np.int32)
        self.op_has_work = np.zeros(self.O, dtype=bool)

        self.station_occupancy = np.zeros((self.K, self.T + 1), dtype=np.int32)
        self.task_occupancy = np.zeros((self.I, self.T + 1), dtype=np.int32)
        self.product_line = np.full(self.P, -1, dtype=np.int32)

        self.y = np.zeros((self.I, self.K), dtype=bool)
        self.u = np.zeros((self.O, self.I), dtype=bool)
        self.x = np.zeros((self.O, self.K, self.I, self.T + 1), dtype=bool)
        self.s = np.zeros(self.I, dtype=np.int32)
        self.e = np.zeros(self.I, dtype=np.int32)
        self.so = np.zeros((self.O, self.I), dtype=np.int32)
        self.eo = np.zeros((self.O, self.I), dtype=np.int32)
        self.a = np.zeros((self.I, self.I), dtype=bool)
        self._refresh_order_matrix()

        self.total_profit = 0.0
        self.total_cost = 0.0
        self.schedule = []
        return self._get_state()

    def _get_state(self):
        """Concatenate dynamic system state vector."""
        return np.concatenate(
            (
                self.completed.astype(np.float32),
                self.discarded.astype(np.float32),
                self.op_times.astype(np.float32) / max(float(self.T), 1.0),
                self.op_locs.astype(np.float32) / max(1.0, float(self.K - 1)),
            )
        )

    def _predecessors_completed(self, task):
        return all(self.completed[pre] for pre in self.Ii_P[task])

    def _line_candidates(self, task, k):
        station_lines = self.station_to_lines[k] if self.station_to_lines[k] else list(range(self.L))
        if not station_lines:
            return []
        p = self.task_to_product[task]
        if p < 0:
            return station_lines
        chosen_line = self.product_line[p]
        if chosen_line == -1:
            return station_lines
        return [chosen_line] if chosen_line in station_lines else []

    def _earliest_precedence_start(self, task):
        if not self.Ii_P[task]:
            return 1
        return max(int(self.task_end_time[pre]) + 1 for pre in self.Ii_P[task])

    def _slot_capacity_ok(self, k, task, start, duration):
        """Vectorized workstation and task capacity boundary verification."""
        end = start + duration
        if np.any(self.station_occupancy[k, start:end] >= self.ck_max[k]):
            return False
        if np.any(self.task_occupancy[task, start:end] >= self.ci_max[task]):
            return False
        return True

    def _has_feasible_assignment(self, task):
        """Verify whether candidate task can be allocated within time budget T."""
        if self.ci_max[task] < 1:
            return False
        pre_start = self._earliest_precedence_start(task)

        for o in range(self.O):
            for k in self.Ko[o]:
                if k in self.KO[o]:
                    continue
                if not self._line_candidates(task, k):
                    continue

                duration = int(self.t_d[k, task])
                if duration <= 0:
                    continue

                move_time = int(self.t_m[self.op_locs[o], k]) if self.op_has_work[o] else 0
                start = max(pre_start, int(self.op_times[o]) + move_time, 1)

                while start + duration <= self.T:
                    if self._slot_capacity_ok(k, task, start, duration):
                        return True
                    start += 1
        return False

    def get_valid_actions(self):
        """
        Action Masking Mechanism (Section 4.1):
        Filters out completed, discarded, precedence-infeasible, or capacity-exceeding tasks.
        Stop action (index I) is always valid.
        """
        valid = []
        for task in range(self.I):
            if self.completed[task] or self.discarded[task]:
                continue
            if not self._has_feasible_assignment(task):
                continue
            valid.append(task)
        valid.append(self.I)
        return valid

    def step(self, action):
        """
        Execute discrete-event transition.

        Returns:
            state (np.ndarray): Next observation vector.
            reward (float): Step net profit (salvage revenue minus operating cost).
            done (bool): Whether terminal conditions are met.
            info (dict): Auxiliary debug information.
        """
        valid_actions = self.get_valid_actions()

        # Penalty for illegal assignments (safety guardrail)
        if action not in valid_actions:
            return self._get_state(), -50.0, True, {"reason": "invalid_action"}

        # Stop action explicitly triggered
        if action == self.I:
            return self._get_state(), 0.0, True, {"reason": "stop"}

        task = int(action)
        pre_start = self._earliest_precedence_start(task)

        best = None
        for o in range(self.O):
            for k in self.Ko[o]:
                if k in self.KO[o]:
                    continue
                candidate_lines = self._line_candidates(task, k)
                if not candidate_lines:
                    continue

                duration = int(self.t_d[k, task])
                if duration <= 0:
                    continue

                move_time = int(self.t_m[self.op_locs[o], k]) if self.op_has_work[o] else 0
                start = max(pre_start, int(self.op_times[o]) + move_time, 1)

                while start + duration <= self.T:
                    if self._slot_capacity_ok(k, task, start, duration):
                        break
                    start += 1

                if start + duration > self.T:
                    continue

                finish = start + duration
                cost = float(self.c[o, k, task]) * duration
                score = (cost, finish, start, o, k, candidate_lines[0])
                if best is None or score < best:
                    best = score

        if best is None:
            self.discarded[task] = True
            reward = -20.0
            done = len(self.get_valid_actions()) == 1
            return self._get_state(), reward, done, {"reason": "no_feasible_assignment"}

        cost, finish, start, o, k, line_choice = best
        duration = int(self.t_d[k, task])

        # Commit schedule occupation
        self.completed[task] = True
        self.task_start_time[task] = start
        self.task_end_time[task] = finish
        self.op_times[o] = finish
        self.op_locs[o] = k
        self.op_has_work[o] = True

        for t in range(start, finish):
            self.station_occupancy[k, t] += 1
            self.task_occupancy[task, t] += 1
            self.x[o, k, task, t] = True

        self.y[task, k] = True
        self.u[o, task] = True
        self.s[task] = start
        self.e[task] = finish
        self.so[:, task] = start
        self.eo[:, task] = finish

        p = self.task_to_product[task]
        if p >= 0 and self.product_line[p] == -1:
            self.product_line[p] = line_choice

        # Automatically discard conflicting tasks
        for conflict in self.Ii_C[task]:
            if not self.completed[conflict]:
                self.discarded[conflict] = True

        self._refresh_order_matrix()
        self.schedule.append((task, int(o), int(k), int(start), int(finish)))

        step_profit = float(self.task_profit[task]) - cost
        self.total_profit += step_profit
        self.total_cost += cost

        # Termination: only stop action remains
        done = len(self.get_valid_actions()) == 1
        return self._get_state(), step_profit, done, {}
