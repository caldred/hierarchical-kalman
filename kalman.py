import numpy as np


class KalmanFilter:
    """
    Kalman filter operations for a single player.

    State: z (intrinsic skills)
    Observation: y = C @ z + noise, where C = H @ B (H selects observed skills)
    """

    def __init__(self, n_skills, B, P_pop=None):
        self.n_skills = n_skills
        self.B = B
        self._I = np.eye(n_skills)
        self.P_pop = P_pop  # Population covariance for uncertainty capping
    
    def predict(self, z_prev, P_prev, Q):
        """Prediction step with asymptotic uncertainty cap at population variance."""
        z_pred = z_prev.copy()

        if self.P_pop is not None:
            P_pred = self._asymptotic_predict_P(P_prev, Q)
        else:
            P_pred = P_prev + Q

        return z_pred, P_pred

    def _asymptotic_predict_P(self, P_prev, Q, min_ratio=0.001):
        """
        Predict uncertainty with asymptotic approach to P_pop.

        P approaches (1 - min_ratio) * P_pop but never exceeds it.
        The rate of approach is driven by the process noise Q.

        Parameters
        ----------
        P_prev : ndarray (n_skills, n_skills)
            Previous covariance matrix
        Q : ndarray (n_skills, n_skills)
            Process noise covariance matrix
        min_ratio : float
            Minimum headroom as fraction of P_pop (default 0.1%)

        Returns
        -------
        P_pred : ndarray (n_skills, n_skills)
            Predicted covariance, capped at (1 - min_ratio) * P_pop
        """
        P_pop = self.P_pop

        # Work in eigenbasis of P_pop for cleaner math
        eigvals_pop, V = np.linalg.eigh(P_pop)

        # Transform to P_pop eigenbasis
        P_prev_basis = V.T @ P_prev @ V
        Q_basis = V.T @ Q @ V

        # For each direction, apply asymptotic update
        p_diag = np.diag(P_prev_basis)
        q_diag = np.diag(Q_basis)
        pop_diag = eigvals_pop

        # Headroom decays exponentially; Q drives the decay
        headroom = pop_diag - p_diag
        # Clamp headroom to be positive (in case P_prev > P_pop initially)
        headroom = np.maximum(headroom, min_ratio * pop_diag)
        decay_rate = q_diag / headroom
        decay = np.exp(-decay_rate)

        # New headroom, but maintain minimum gap
        min_headroom = min_ratio * pop_diag
        new_headroom = np.maximum(headroom * decay, min_headroom)

        p_new_diag = pop_diag - new_headroom

        # Reconstruct full matrix
        P_pred = V @ np.diag(p_new_diag) @ V.T
        return 0.5 * (P_pred + P_pred.T)  # Symmetrize
    
    def _update_core(self, z_pred, P_pred, y, obs_indices, R_diag):
        """
        Core update computation.
        
        Uses Joseph form for numerical stability:
        P = (I - KC) P (I - KC)' + K R K'
        
        Returns
        -------
        z_upd : ndarray
            Updated state
        P_upd : ndarray
            Updated covariance
        v : ndarray
            Innovation (y - y_pred)
        S : ndarray
            Innovation covariance
        """
        C = self.B[obs_indices, :]
        R = np.diag(R_diag)
        
        # Innovation
        y_pred = C @ z_pred
        v = y - y_pred
        
        # Innovation covariance
        S = C @ P_pred @ C.T + R
        
        # Kalman gain via solve for stability
        try:
            K = np.linalg.solve(S.T, C @ P_pred.T).T
        except np.linalg.LinAlgError:
            K = P_pred @ C.T @ np.linalg.pinv(S)
        
        # State update
        z_upd = z_pred + K @ v
        
        # Covariance update using Joseph form
        IKC = self._I - K @ C
        P_upd = IKC @ P_pred @ IKC.T + K @ R @ K.T
        
        # Symmetrize (floating point)
        P_upd = 0.5 * (P_upd + P_upd.T)
        
        return z_upd, P_upd, v, S
    
    def update(self, z_pred, P_pred, y, observed_mask, R_diag):
        """
        Update step with partial observations.
        
        Parameters
        ----------
        z_pred : ndarray
            Predicted state
        P_pred : ndarray
            Predicted covariance
        y : ndarray
            Observed values (only for observed skills)
        observed_mask : ndarray
            Boolean mask of which skills are observed
        R_diag : ndarray
            Observation noise variances for observed skills
        
        Returns
        -------
        z_upd : ndarray
            Updated state
        P_upd : ndarray
            Updated covariance
        """
        if not observed_mask.any():
            return z_pred.copy(), P_pred.copy()
        
        obs_indices = np.where(observed_mask)[0]
        z_upd, P_upd, _, _ = self._update_core(z_pred, P_pred, y, obs_indices, R_diag)
        return z_upd, P_upd
    
    def filter_sequence(self, observations, Q, R_base, z0, P0):
        """
        Run filter over a sequence of time bins for one player.
        
        Parameters
        ----------
        observations : list of dict
            Each dict has keys 'mask' (bool array), 'y' (observed values), 
            'n_obs' (observation counts)
        Q : ndarray
            (n_skills,) process noise variances
        R_base : ndarray
            (n_skills,) base observation noise variances
        z0 : ndarray
            Initial state
        P0 : ndarray
            Initial covariance
        
        Returns
        -------
        z_filt : ndarray (T, n_skills)
            Filtered states
        P_filt : ndarray (T, n_skills, n_skills)
            Filtered covariances
        innovations : list
            Innovation vectors (None for bins with no observations)
        S_list : list
            Innovation covariances (None for bins with no observations)
        """
        T = len(observations)
        Q_mat = np.diag(Q)
        
        z_filt = np.zeros((T, self.n_skills))
        P_filt = np.zeros((T, self.n_skills, self.n_skills))
        innovations = []
        S_list = []
        
        z_prev, P_prev = z0.copy(), P0.copy()
        
        for t in range(T):
            z_pred, P_pred = self.predict(z_prev, P_prev, Q_mat)
            
            obs = observations[t]
            mask = obs['mask']
            
            if mask.any():
                y = obs['y']
                n_obs = obs['n_obs']
                obs_indices = np.where(mask)[0]
                
                # Scale observation noise by sample size
                R_diag = R_base[obs_indices] / np.maximum(n_obs, 1)
                
                # Single call for update and diagnostics
                z_upd, P_upd, v, S = self._update_core(
                    z_pred, P_pred, y, obs_indices, R_diag
                )
                innovations.append(v)
                S_list.append(S)
            else:
                z_upd, P_upd = z_pred.copy(), P_pred.copy()
                innovations.append(None)
                S_list.append(None)
            
            z_filt[t] = z_upd
            P_filt[t] = P_upd
            z_prev, P_prev = z_upd, P_upd
        
        return z_filt, P_filt, innovations, S_list
    
    def smooth_sequence(self, z_filt, P_filt, Q):
        """
        Rauch-Tung-Striebel smoother.
        
        Parameters
        ----------
        z_filt : ndarray (T, n_skills)
            Filtered states from forward pass
        P_filt : ndarray (T, n_skills, n_skills)
            Filtered covariances from forward pass
        Q : ndarray
            (n_skills,) process noise variances
        
        Returns
        -------
        z_smooth : ndarray (T, n_skills)
            Smoothed states
        P_smooth : ndarray (T, n_skills, n_skills)
            Smoothed covariances
        P_cross : ndarray (T-1, n_skills, n_skills)
            Cross-covariances E[z_t z_{t+1}'] for EM
        """
        T = z_filt.shape[0]
        Q_mat = np.diag(Q)
        
        z_smooth = np.zeros_like(z_filt)
        P_smooth = np.zeros_like(P_filt)
        P_cross = np.zeros((T - 1, self.n_skills, self.n_skills))
        
        z_smooth[-1] = z_filt[-1]
        P_smooth[-1] = P_filt[-1]
        
        for t in range(T - 2, -1, -1):
            # Predict covariance with asymptotic cap if P_pop available
            if self.P_pop is not None:
                P_pred = self._asymptotic_predict_P(P_filt[t], Q_mat)
            else:
                P_pred = P_filt[t] + Q_mat

            # Smoother gain
            try:
                J = np.linalg.solve(P_pred.T, P_filt[t].T).T
            except np.linalg.LinAlgError:
                J = P_filt[t] @ np.linalg.pinv(P_pred)
            
            z_smooth[t] = z_filt[t] + J @ (z_smooth[t + 1] - z_filt[t])
            P_smooth[t] = P_filt[t] + J @ (P_smooth[t + 1] - P_pred) @ J.T
            P_smooth[t] = 0.5 * (P_smooth[t] + P_smooth[t].T)
            
            # Cross-covariance for EM
            P_cross[t] = J @ P_smooth[t + 1]
        
        return z_smooth, P_smooth, P_cross
    
    def get_skill_estimates(self, z, P):
        """
        Convert intrinsic state estimates to actual skill estimates.
        
        s = B @ z
        Var(s) = diag(B @ P @ B')
        
        Parameters
        ----------
        z : ndarray
            Intrinsic state, shape (n_skills,) or (T, n_skills)
        P : ndarray
            State covariance, shape (n_skills, n_skills) or (T, n_skills, n_skills)
        
        Returns
        -------
        s : ndarray
            Skill estimates, same leading shape as z
        s_var : ndarray
            Skill variances, same leading shape as z
        """
        if z.ndim == 1:
            s = self.B @ z
            s_var = np.diag(self.B @ P @ self.B.T)
        else:
            # Batch: z is (T, n_skills)
            s = (self.B @ z.T).T
            # Diagonal of B @ P[t] @ B.T for each t
            s_var = np.einsum('ij,tjk,ik->ti', self.B, P, self.B)
        
        return s, s_var