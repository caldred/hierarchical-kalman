import numpy as np
import polars as pl
import pickle
from dataclasses import dataclass, field
from joblib import Parallel, delayed

from dag import DAG
from kalman import KalmanFilter
from parameters import ParameterEstimator
from preprocessing import (
    create_time_bins,
    compute_population_prior
)


# ---------------------------------------------------------------------------
# Standalone functions for parallelization (must be picklable)
# ---------------------------------------------------------------------------

def _filter_single_player(player_id, obs_seq, kf, Q, R, z0, P0):
    """Filter a single player."""
    z_filt, P_filt, innovations, S_list = kf.filter_sequence(
        obs_seq, Q, R, z0, P0
    )
    return player_id, z_filt, P_filt, innovations, S_list


def _smooth_single_player(player_id, z_filt, P_filt, kf, Q):
    """Smooth a single player."""
    z_smooth, P_smooth, P_cross = kf.smooth_sequence(z_filt, P_filt, Q)
    return player_id, z_smooth, P_smooth, P_cross


def _final_filter_single_player(player_id, obs_seq, kf, Q, R, z0, P0):
    """Final filter for a single player."""
    z_filt, P_filt, _, _ = kf.filter_sequence(obs_seq, Q, R, z0, P0)
    return player_id, z_filt, P_filt


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EMConfig:
    """Configuration for EM algorithm behavior."""
    max_iter: int = 20
    min_iter: int = 3
    tol: float = 1e-3
    abs_tol: float = 20.0
    tol_theta: float = 1e-4
    use_aitken: bool = True
    verbose: bool = True


@dataclass
class DampingConfig:
    """Configuration for parameter update damping."""
    alpha: float = 0.99
    alpha_min: float = 0.98
    alpha_max: float = 1.0
    max_spectral_radius: float = 0.95


@dataclass
class InitConfig:
    """Configuration for parameter initialization."""
    w_init_scale: float = 0.1
    q_scale: float = 0.03
    r_init: float = 0.5


@dataclass
class EMState:
    """Mutable state tracked across EM iterations."""
    ll_hist: list = field(default_factory=list)
    theta_prev: np.ndarray = None
    iteration: int = 0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class HierarchicalKalmanFilter:
    """
    Hierarchical Kalman Filter with DAG structure for baseball player skills.

    State: z (intrinsic skills, random walk)
    Observation: y = H @ s + noise, where s = B @ z (actual skills)
    """

    def __init__(
        self,
        dag,
        bin_size_days=7,
        n_jobs=-1
    ):
        """
        Parameters
        ----------
        dag : dict
            Mapping child skill -> list of parent skills
        bin_size_days : int
            Time bin width in days
        n_jobs : int
            Number of parallel jobs (-1 for all cores)
        """
        self.dag_spec = dag
        self.bin_size_days = bin_size_days
        self.n_jobs = n_jobs

        # Set during fit
        self.dag = None
        self.skill_names = None
        self.skill_to_idx = None
        self.n_skills = None

        # Learned parameters
        self.W = None
        self.B = None
        self.Q = None
        self.R = None
        self.mu_pop = None
        self.sigma_pop = None
        self.P_pop = None  # Population covariance in intrinsic space (for uncertainty capping)

        # State tracking
        self.player_states = {}
        self.time_bins = None
        self.origin_date = None
        self.player_start_bins = {}

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def fit(
        self,
        df,
        max_iter=20,
        tol=1e-3,
        abs_tol=20,
        verbose=True,
    ):
        """
        Fit model parameters using EM algorithm.

        Parameters
        ----------
        df : polars.DataFrame
            Columns: [player_id, timestamp, skill_name, observed_value]
        max_iter : int
            Maximum EM iterations (overridden by em_config if provided)
        tol : float
            Convergence tolerance (overridden by em_config if provided)
        abs_tol : float
            Absolute tolerance (overridden by em_config if provided)
        verbose : bool
            Print progress (overridden by em_config if provided)
        em_config : EMConfig, optional
            Full EM configuration (overrides individual params)
        damping_config : DampingConfig, optional
            Damping configuration
        init_config : InitConfig, optional
            Initialization configuration
        """
        # Build configs
        em_config = EMConfig(
            max_iter=max_iter,
            tol=tol,
            abs_tol=abs_tol,
            verbose=verbose
        )

        damping_config = DampingConfig()

        init_config = InitConfig()

        # Initialize model structure and parameters
        self._initialize_from_data(df, init_config, em_config.verbose)

        # Prepare binned observations
        binned_data, player_ids = self._prepare_binned_data(df, em_config.verbose)

        # Count effective dimensions for convergence scaling
        n_eff = self._count_effective_dims(binned_data)
        if em_config.verbose:
            print(f"Effective LL dims (n_eff): {n_eff}")

        # Run EM
        param_estimator = ParameterEstimator(self.dag, self.n_skills)
        em_state = EMState()

        for iteration in range(em_config.max_iter):
            em_state.iteration = iteration

            # E-step
            e_step_results = self._e_step(binned_data, player_ids)

            # Compute and record log-likelihood
            ll = param_estimator.compute_log_likelihood(
                e_step_results['innovations'],
                e_step_results['S']
            )
            em_state.ll_hist.append(ll)

            if em_config.verbose:
                print(f"Iteration {iteration + 1}: LL = {ll:.2f}")

            # Check convergence
            converged = self._check_convergence(em_state, n_eff, em_config)
            if converged:
                break

            # M-step
            self._m_step(
                binned_data,
                e_step_results,
                param_estimator,
                df,
                em_state,
                damping_config
            )

        # Store final filtered states
        self._store_final_states(binned_data, player_ids)

        if em_config.verbose:
            self._print_learned_params()

        return self

    def get_estimates(self, player_ids=None):
        """
        Get current skill estimates for all or specified players.

        Returns polars DataFrame with columns:
        [player_id, skill_name, mean, std, time_bin]
        """
        if player_ids is None:
            player_ids = list(self.player_states.keys())

        rows = []
        for player_id in player_ids:
            if player_id not in self.player_states:
                continue

            state = self.player_states[player_id]
            z = state['z']
            P = state['P']
            bin_idx = state['last_bin']

            s = self.B @ z
            s_var = np.diag(self.B @ P @ self.B.T)

            for i, skill_name in enumerate(self.skill_names):
                rows.append({
                    'player_id': player_id,
                    'skill_name': skill_name,
                    'mean': s[i],
                    'std': np.sqrt(s_var[i]),
                    'time_bin': bin_idx
                })

        return pl.DataFrame(rows)

    def get_historical_estimates(self, df):
        """
        Get filtered estimates for all historical time bins.

        Returns polars DataFrame with columns:
        [player_id, time_bin, bin_start, skill_name, mean, std]
        """
        binned_data, time_bins, _, player_start_bins = create_time_bins(
            df, self.bin_size_days, self.skill_names,
            origin_date=self.origin_date
        )

        z0, P0 = self._intrinsic_prior()
        kf = KalmanFilter(self.n_skills, self.B, P_pop=self.P_pop)

        rows = []
        for player_id, obs_seq in binned_data.items():
            start_bin = player_start_bins.get(player_id, 0)
            z_filt, P_filt, _, _ = kf.filter_sequence(
                obs_seq, self.Q, self.R, z0, P0
            )

            s_filt, s_var = kf.get_skill_estimates(z_filt, P_filt)

            for t in range(len(obs_seq)):
                abs_bin = start_bin + t
                if abs_bin >= len(time_bins):
                    break
                bin_start = time_bins[abs_bin][0]
                for i, skill_name in enumerate(self.skill_names):
                    rows.append({
                        'player_id': player_id,
                        'time_bin': abs_bin,
                        'bin_start': bin_start,
                        'skill_name': skill_name,
                        'mean': s_filt[t, i],
                        'std': np.sqrt(s_var[t, i])
                    })

        return pl.DataFrame(rows)

    def update(self, new_df):
        """
        Incremental update with new observations.

        Uses stored origin_date to ensure consistent bin indexing.
        """
        if self.B is None:
            raise ValueError("Model must be fit before update")

        if self.origin_date is None:
            raise ValueError("Model origin_date not set. Was fit() called?")

        new_df = new_df.with_columns(
            pl.col('timestamp').cast(pl.Datetime).alias('timestamp')
        )
        new_max_date = new_df.select(pl.col('timestamp').max()).item()

        binned_data, new_time_bins, _, new_start_bins = create_time_bins(
            new_df, self.bin_size_days, self.skill_names,
            origin_date=self.origin_date,
            end_date=new_max_date
        )

        z0, P0 = self._intrinsic_prior()
        kf = KalmanFilter(self.n_skills, self.B, P_pop=self.P_pop)

        for player_id, obs_seq in binned_data.items():
            start_bin = new_start_bins.get(player_id, 0)
            end_bin = start_bin + len(obs_seq) - 1

            if player_id in self.player_states:
                state = self.player_states[player_id]
                z_prev = state['z']
                P_prev = state['P']
                last_processed_bin = state['last_bin']

                next_abs_bin = last_processed_bin + 1
                if next_abs_bin > end_bin:
                    continue

                local_start = max(0, next_abs_bin - start_bin)
                new_obs_seq = obs_seq[local_start:]

                z_filt, P_filt, _, _ = kf.filter_sequence(
                    new_obs_seq, self.Q, self.R, z_prev, P_prev
                )

                self.player_states[player_id] = {
                    'z': z_filt[-1],
                    'P': P_filt[-1],
                    'last_bin': start_bin + local_start + len(new_obs_seq) - 1
                }
            else:
                z_filt, P_filt, _, _ = kf.filter_sequence(
                    obs_seq, self.Q, self.R, z0, P0
                )

                self.player_states[player_id] = {
                    'z': z_filt[-1],
                    'P': P_filt[-1],
                    'last_bin': start_bin + len(obs_seq) - 1
                }
                if player_id not in self.player_start_bins:
                    self.player_start_bins[player_id] = start_bin

        if new_time_bins and (self.time_bins is None or len(new_time_bins) > len(self.time_bins)):
            self.time_bins = new_time_bins

    def predict_skill(self, player_id, skill_name):
        """
        Get current estimate for a specific player and skill.

        Returns (mean, std).
        """
        if player_id not in self.player_states:
            idx = self.skill_to_idx[skill_name]
            return self.mu_pop[idx], self.sigma_pop[idx]

        state = self.player_states[player_id]
        z = state['z']
        P = state['P']

        s = self.B @ z
        s_var = np.diag(self.B @ P @ self.B.T)

        idx = self.skill_to_idx[skill_name]
        return s[idx], np.sqrt(s_var[idx])

    def predict_forward(self, player_id, n_bins=1):
        """
        Predict player skills n_bins into the future.

        Returns dict of skill_name -> (mean, std).
        """
        if player_id not in self.player_states:
            return {
                name: (self.mu_pop[i], self.sigma_pop[i])
                for i, name in enumerate(self.skill_names)
            }

        state = self.player_states[player_id]
        z = state['z'].copy()
        P = state['P'].copy()

        Q_mat = np.diag(self.Q)

        for _ in range(n_bins):
            P = P + Q_mat

        s = self.B @ z
        s_var = np.diag(self.B @ P @ self.B.T)

        return {
            name: (s[i], np.sqrt(s_var[i]))
            for i, name in enumerate(self.skill_names)
        }

    def save(self, path):
        """Save model to file."""
        with open(path, 'wb') as f:
            pickle.dump(self._to_dict(), f)

    @classmethod
    def load(cls, path):
        """Load model from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return cls._from_dict(data)

    # -----------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------

    def _initialize_from_data(self, df, init_config, verbose):
        """Initialize skill names, DAG, population prior, and parameters."""
        # Infer skill names from DAG and data
        dag_skills = set()
        for child, parents in self.dag_spec.items():
            dag_skills.add(child)
            dag_skills.update(parents)

        data_skills = set(
            df.select(pl.col('skill_name').unique()).to_series().to_list()
        )
        self.skill_names = sorted(dag_skills | data_skills)
        self.n_skills = len(self.skill_names)
        self.skill_to_idx = {name: i for i, name in enumerate(self.skill_names)}

        self.dag = DAG(self.dag_spec, self.skill_names)

        # Population prior
        self.mu_pop, self.sigma_pop = compute_population_prior(df, self.skill_names)

        # Initialize parameters
        self.W = self.dag.init_weight_matrix(init_scale=init_config.w_init_scale)
        self.B = self.dag.compute_B_matrix(self.W)
        self.Q = (self.sigma_pop * init_config.q_scale) ** 2
        self.R = np.ones(self.n_skills) * init_config.r_init

    def _prepare_binned_data(self, df, verbose):
        """Create time bins and return binned observations."""
        result = create_time_bins(
            df,
            self.bin_size_days,
            self.skill_names,
            origin_date=None
        )
        binned_data, self.time_bins, self.origin_date, self.player_start_bins = result

        player_ids = sorted(binned_data.keys())
        n_bins = len(self.time_bins)
        n_players = len(player_ids)

        if verbose:
            print(f"Data: {n_players} players, {n_bins} time bins, {self.n_skills} skills")
            print(f"Origin date: {self.origin_date}")
            if n_players > 0:
                seq_lengths = [len(binned_data[pid]) for pid in player_ids]
                min_start = min(self.player_start_bins.values())
                max_start = max(self.player_start_bins.values())
                print(
                    f"Player trimming: start_bin range={min_start}-{max_start}, "
                    f"seq_len min/avg/max={min(seq_lengths)}/{np.mean(seq_lengths):.1f}/{max(seq_lengths)}"
                )

        return binned_data, player_ids

    def _count_effective_dims(self, binned_data):
        """Count total observed dimensions across all players and time steps."""
        n_eff = 0
        for pid in binned_data:
            for obs in binned_data[pid]:
                n_eff += int(np.sum(obs['mask']))
        return max(1, n_eff)

    # -----------------------------------------------------------------------
    # E-step
    # -----------------------------------------------------------------------

    def _e_step(self, binned_data, player_ids):
        """
        Run E-step: filter and smooth all players.

        Returns dict with keys:
            z_smooth, P_smooth, P_cross, innovations, S, z_filt, P_filt
        """
        z0, P0 = self._intrinsic_prior()
        kf = KalmanFilter(self.n_skills, self.B, P_pop=self.P_pop)

        # Filter
        filter_results = Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(_filter_single_player)(
                pid, binned_data[pid], kf, self.Q, self.R, z0, P0
            )
            for pid in player_ids
        )

        z_filt_all = {}
        P_filt_all = {}
        innovations_all = {}
        S_all = {}

        for pid, z_filt, P_filt, innovations, S_list in filter_results:
            z_filt_all[pid] = z_filt
            P_filt_all[pid] = P_filt
            innovations_all[pid] = innovations
            S_all[pid] = S_list

        # Smooth
        smooth_results = Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(_smooth_single_player)(
                pid, z_filt_all[pid], P_filt_all[pid], kf, self.Q
            )
            for pid in player_ids
        )

        z_smooth_all = {}
        P_smooth_all = {}
        P_cross_all = {}

        for pid, z_smooth, P_smooth, P_cross in smooth_results:
            z_smooth_all[pid] = z_smooth
            P_smooth_all[pid] = P_smooth
            P_cross_all[pid] = P_cross

        return {
            'z_smooth': z_smooth_all,
            'P_smooth': P_smooth_all,
            'P_cross': P_cross_all,
            'innovations': innovations_all,
            'S': S_all,
            'z_filt': z_filt_all,
            'P_filt': P_filt_all
        }

    # -----------------------------------------------------------------------
    # M-step
    # -----------------------------------------------------------------------

    def _m_step(
        self,
        binned_data,
        e_step_results,
        param_estimator,
        df,
        em_state,
        damping_config
    ):
        """Update parameters W, Q, R with damping."""
        z_smooth_all = e_step_results['z_smooth']
        P_smooth_all = e_step_results['P_smooth']
        P_cross_all = e_step_results['P_cross']

        edge_mask = self.dag.get_edge_mask()

        # Store previous values for damping
        W_prev = self.W.copy()
        Q_prev = self.Q.copy()
        R_prev = self.R.copy()

        # Estimate new parameters
        W_new = param_estimator.estimate_edge_weights(
            binned_data, z_smooth_all, self.B, n_jobs=self.n_jobs
        )
        W_new = W_new * edge_mask

        # Temporarily update B for Q/R estimation
        self.W = W_new
        self.B = self.dag.compute_B_matrix(self.W)

        Q_new = param_estimator.estimate_process_noise(
            z_smooth_all,
            P_smooth_all,
            P_cross_all
        )

        R_new = param_estimator.estimate_observation_noise(
            z_smooth_all,
            self.B,
            df,
            self.skill_names,
            self.bin_size_days,
            self.origin_date,
            self.player_start_bins
        )

        # Compute damping factor
        alpha = self._compute_damping_alpha(em_state, damping_config)

        # Apply damped updates
        self.W = (1.0 - alpha) * W_prev + alpha * W_new
        self.Q = (1.0 - alpha) * Q_prev + alpha * Q_new
        self.R = (1.0 - alpha) * R_prev + alpha * R_new

        # Clamp spectral radius for stability
        self.W = self._clamp_spectral_radius(self.W, damping_config.max_spectral_radius)
        self.B = self.dag.compute_B_matrix(self.W)

    def _compute_damping_alpha(self, em_state, damping_config):
        """Compute damping factor based on LL trajectory."""
        if em_state.iteration < 1:
            return damping_config.alpha

        delta_ll = em_state.ll_hist[-1] - em_state.ll_hist[-2]

        if delta_ll < 0:
            return max(damping_config.alpha_min, damping_config.alpha * 0.5)
        else:
            return min(damping_config.alpha_max, damping_config.alpha)

    @staticmethod
    def _clamp_spectral_radius(W, max_rho):
        """Scale W if spectral radius exceeds threshold."""
        eigvals = np.linalg.eigvals(W)
        rho = float(np.max(np.abs(eigvals))) if eigvals.size > 0 else 0.0
        if rho > max_rho and np.isfinite(rho):
            return W * (max_rho / rho)
        return W

    # -----------------------------------------------------------------------
    # Convergence
    # -----------------------------------------------------------------------

    def _check_convergence(self, em_state, n_eff, em_config):
        """
        Check convergence criteria.

        Returns True if converged.
        """
        if em_state.iteration < 1:
            em_state.theta_prev = self._pack_theta()
            return False

        ll_hist = em_state.ll_hist
        delta_ll = ll_hist[-1] - ll_hist[-2]
        delta_per_dim = delta_ll / n_eff

        # Parameter movement
        theta = self._pack_theta()
        dtheta_rel = self._rel_change(theta, em_state.theta_prev)

        # Aitken acceleration
        aitken_gap_per_dim = None
        if em_config.use_aitken and em_state.iteration >= 2:
            aitken_gap_per_dim = self._compute_aitken_gap(ll_hist, n_eff)

        if em_config.verbose:
            msg = f"  delta_LL={delta_ll:.2f}, per_dim={delta_per_dim:.3e}, dtheta_rel={dtheta_rel:.3e}"
            if aitken_gap_per_dim is not None:
                msg += f", aitken_gap_per_dim={aitken_gap_per_dim:.3e}"
            print(msg)

        em_state.theta_prev = theta

        # Check stopping conditions
        if em_state.iteration + 1 < em_config.min_iter:
            return False

        stop_ll = (delta_per_dim < em_config.tol) or (abs(delta_ll) < em_config.abs_tol)
        stop_theta = (dtheta_rel < em_config.tol_theta)
        stop_aitken = (aitken_gap_per_dim is not None) and (aitken_gap_per_dim < em_config.tol)

        if (stop_ll or stop_aitken) and stop_theta:
            if em_config.verbose:
                print(
                    f"Converged after {em_state.iteration + 1} iterations "
                    f"(per_dim={delta_per_dim:.3e}, dtheta_rel={dtheta_rel:.3e})"
                )
            return True

        return False

    def _pack_theta(self):
        """Pack parameters into a single vector for convergence checking."""
        edge_mask = self.dag.get_edge_mask()
        w_flat = (self.W * edge_mask).ravel()
        q_flat = np.asarray(self.Q).ravel()
        r_flat = np.asarray(self.R).ravel()
        return np.concatenate([w_flat, q_flat, r_flat])

    @staticmethod
    def _rel_change(x, x_prev):
        """Compute relative change between parameter vectors."""
        num = float(np.linalg.norm(x - x_prev))
        den = max(1.0, float(np.linalg.norm(x_prev)))
        return num / den

    @staticmethod
    def _compute_aitken_gap(ll_hist, n_eff):
        """Compute Aitken acceleration estimate of remaining gap."""
        ll_k = ll_hist[-1]
        ll_km1 = ll_hist[-2]
        ll_km2 = ll_hist[-3]

        d1 = ll_k - ll_km1
        d0 = ll_km1 - ll_km2

        if abs(d0) <= 0:
            return None

        a = d1 / d0
        if abs(1.0 - a) <= 1e-12:
            return None

        ll_inf = ll_km1 + d1 / (1.0 - a)
        return abs(ll_inf - ll_k) / n_eff

    # -----------------------------------------------------------------------
    # Final state storage
    # -----------------------------------------------------------------------

    def _store_final_states(self, binned_data, player_ids):
        """Run final filter pass and store terminal states."""
        z0, P0 = self._intrinsic_prior()
        kf = KalmanFilter(self.n_skills, self.B, P_pop=self.P_pop)

        final_results = Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(_final_filter_single_player)(
                pid, binned_data[pid], kf, self.Q, self.R, z0, P0
            )
            for pid in player_ids
        )

        for pid, z_filt, P_filt in final_results:
            start_bin = self.player_start_bins.get(pid, 0)
            self.player_states[pid] = {
                'z': z_filt[-1],
                'P': P_filt[-1],
                'last_bin': start_bin + z_filt.shape[0] - 1
            }

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _intrinsic_prior(self):
        """Compute prior in intrinsic (z) space from population stats."""
        if self.B is None:
            raise ValueError("B matrix is not initialized")

        inv_B = np.linalg.solve(self.B, np.eye(self.n_skills))
        z0 = inv_B @ self.mu_pop
        P0 = inv_B @ np.diag(self.sigma_pop ** 2) @ inv_B.T

        # Store P_pop for uncertainty capping (P should never exceed population variance)
        self.P_pop = P0.copy()

        return z0, P0

    def _format_edge_weights(self):
        """Format edge weights for display."""
        lines = []
        for i, child in enumerate(self.skill_names):
            parents = self.dag.get_parents(child)
            if parents:
                weights = {p: round(self.W[i, self.skill_to_idx[p]], 4) for p in parents}
                lines.append(f"    {child} <- {weights}")
        return "\n".join(lines) if lines else "    (no edges)"

    def _print_learned_params(self):
        """Print learned parameters."""
        print("\nLearned parameters:")
        print(f"  Edge weights (W):\n{self._format_edge_weights()}")
        print(f"  Process noise (sqrt Q): {dict(zip(self.skill_names, np.sqrt(self.Q).round(4)))}")
        print(f"  Observation noise (sqrt R): {dict(zip(self.skill_names, np.sqrt(self.R).round(4)))}")

    # -----------------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------------

    def _to_dict(self):
        """Convert model to dict for serialization."""
        return {
            'dag_spec': self.dag_spec,
            'bin_size_days': self.bin_size_days,
            'skill_names': self.skill_names,
            'W': self.W,
            'B': self.B,
            'Q': self.Q,
            'R': self.R,
            'mu_pop': self.mu_pop,
            'sigma_pop': self.sigma_pop,
            'P_pop': self.P_pop,
            'player_states': self.player_states,
            'origin_date': self.origin_date,
            'time_bins': self.time_bins,
            'player_start_bins': self.player_start_bins
        }

    @classmethod
    def _from_dict(cls, data):
        """Reconstruct model from dict."""
        model = cls(data['dag_spec'], data['bin_size_days'])
        model.skill_names = data['skill_names']
        model.n_skills = len(model.skill_names)
        model.skill_to_idx = {name: i for i, name in enumerate(model.skill_names)}
        model.dag = DAG(data['dag_spec'], model.skill_names)
        model.W = data['W']
        model.B = data['B']
        model.Q = data['Q']
        model.R = data['R']
        model.mu_pop = data['mu_pop']
        model.sigma_pop = data['sigma_pop']
        model.P_pop = data.get('P_pop')  # May be None for older serialized models
        model.player_states = data['player_states']
        model.origin_date = data.get('origin_date')
        model.time_bins = data.get('time_bins')
        model.player_start_bins = data.get('player_start_bins', {})

        return model
