/*
 * EdgeEngine AWARE - node-side runtime (see eea_node.h).
 *
 * Every function mirrors a Python function of the reference implementation;
 * the Python name is given in the comment above it. Keep the two in sync.
 */
#include "eea_node.h"

#include <math.h>
#include <string.h>

#define EEA_PI 3.14159265358979323846
#define EEA_DAY_S 86400.0
#define EEA_N_PRIORITY_LEVELS 3

static eea_real eea_max(eea_real a, eea_real b) { return a > b ? a : b; }
static eea_real eea_min(eea_real a, eea_real b) { return a < b ? a : b; }
static float eea_clipf(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }

/* NodeProfile.path_loss_from_margin */
static eea_real eea_path_loss_from_margin(const eea_profile_t *p, int mode, eea_real margin_db) {
    return p->tx_power_dbm[mode] - p->sensitivity_dbm[mode] - margin_db;
}

/* NodeProfile.margin_for_mode */
static eea_real eea_margin_for_mode(const eea_profile_t *p, int mode, eea_real path_loss_db) {
    return p->tx_power_dbm[mode] - path_loss_db - p->sensitivity_dbm[mode];
}

/* ------------------------------------------------------------------------
 * Tracker
 * ---------------------------------------------------------------------- */
void eea_tracker_init(eea_tracker_t *t, const eea_profile_t *profile) {
    memset(t, 0, sizeof(*t));
    t->profile = profile;
    eea_tracker_reset(t);
}

/* NodeStateTracker.reset */
void eea_tracker_reset(eea_tracker_t *t) {
    const eea_profile_t *p = t->profile;
    memset(t, 0, sizeof(*t));
    t->profile = p;
    t->capacity_j = 1.0;
    t->priority = 0;
    t->link_quality = 1.0;
}

/* NodeStateTracker.begin_step */
void eea_tracker_begin_step(eea_tracker_t *t, eea_real now_s, eea_real time_of_day_s, eea_real energy_j,
                            eea_real capacity_j, eea_real harvest_power_w, int priority) {
    const eea_real alpha = t->profile->harvest_ewma_alpha;
    t->now_s = now_s;
    t->time_of_day_s = time_of_day_s;
    t->energy_j = energy_j;
    t->capacity_j = capacity_j;
    t->harvest_w = eea_max(0.0, harvest_power_w);
    if (!t->harvest_initialised) {
        t->harvest_recent_w = t->harvest_w;
        t->harvest_initialised = true;
    } else {
        t->harvest_recent_w = (1 - alpha) * t->harvest_recent_w + alpha * t->harvest_w;
    }
    if (priority >= 0) t->priority = priority;
}

/* NodeStateTracker.on_measurement */
void eea_tracker_on_measurement(eea_tracker_t *t, const eea_measurement_t *m) {
    t->measurement = *m;
    t->has_measurement = true;
}

/* NodeStateTracker.on_transmission - the packet carries the stored measurement */
void eea_tracker_on_transmission(eea_tracker_t *t, eea_real sent_at_s, int acked, eea_real now_s, int mode,
                                 bool has_margin, eea_real margin_db) {
    const eea_profile_t *p = t->profile;
    if (acked < 0 || !p->ack_available) {
        t->has_ack = true;
        t->last_ack_time_s = now_s;
        t->last_ack_measurement = t->measurement;
        t->last_ack_sent_at_s = sent_at_s;
        return;
    }
    const eea_real alpha = p->link_ewma_alpha;
    t->link_quality = (1 - alpha) * t->link_quality + alpha * (acked ? 1.0 : 0.0);
    if (acked) {
        t->has_ack = true;
        t->last_ack_time_s = now_s;
        t->last_ack_measurement = t->measurement;
        t->last_ack_sent_at_s = sent_at_s;
        if (has_margin) {
            t->path_loss_est_db = eea_path_loss_from_margin(p, mode, margin_db);
            t->has_path_loss_est = true;
        }
    } else {
        const eea_real floor_db = eea_path_loss_from_margin(p, mode, 0.0);
        t->path_loss_est_db = t->has_path_loss_est ? eea_max(t->path_loss_est_db, floor_db) : floor_db;
        t->has_path_loss_est = true;
    }
}

/* NodeStateTracker.set_priority */
void eea_tracker_set_priority(eea_tracker_t *t, int priority) { t->priority = priority; }

/* NodeStateTracker.state */
void eea_tracker_state(const eea_tracker_t *t, eea_node_state_t *s) {
    const eea_profile_t *p = t->profile;
    const eea_real age_cap = p->age_scale_s;
    memset(s, 0, sizeof(*s));
    if (!t->has_measurement) {
        s->has_measurement = false;
        s->measurement_value = 0.0;
        s->measurement_noise_std = 0.0;
        s->measurement_age_s = age_cap;
    } else {
        s->has_measurement = true;
        s->measurement_value = t->measurement.value;
        s->measurement_noise_std = t->measurement.noise_std;
        s->measurement_age_s = t->now_s - t->measurement.timestamp_s;
    }
    if (!t->has_ack) {
        s->has_reported = false;
        s->time_since_tx_success_s = age_cap;
        s->app_info_age_s = age_cap;
        s->reported_value = 0.0;
    } else {
        s->has_reported = true;
        s->time_since_tx_success_s = t->now_s - t->last_ack_time_s;
        /* Packet.measurement_age_s = sent_at - measurement.timestamp */
        s->app_info_age_s = s->time_since_tx_success_s + (t->last_ack_sent_at_s - t->last_ack_measurement.timestamp_s);
        s->reported_value = t->last_ack_measurement.value;
    }
    s->time_of_day_s = t->time_of_day_s;
    s->stored_energy_j = t->energy_j;
    s->capacity_j = t->capacity_j;
    s->harvest_power_w = t->harvest_w;
    s->harvest_power_recent_w = t->harvest_recent_w;
    s->app_priority = t->priority;
    s->link_quality = t->link_quality;
    s->has_link_estimate = t->has_path_loss_est;
    s->path_loss_est_db = t->has_path_loss_est ? t->path_loss_est_db : p->path_loss_max_db;
    s->sensing_energy_low_j = p->sensing_energy_j[1];
    s->sensing_energy_high_j = p->sensing_energy_j[2];
    s->tx_energy_j = p->tx_energy_j[p->reference_mode];
}

/* ------------------------------------------------------------------------
 * Observation
 * ---------------------------------------------------------------------- */
/* ObservationBuilder.importance */
static eea_real eea_importance(const eea_profile_t *p, eea_real value, bool has_measurement) {
    if (!has_measurement) return 0.0;
    if (value <= p->critical_threshold) return 1.0;
    const eea_real dist = eea_min(fabs(value - p->warning_threshold), fabs(value - p->critical_threshold));
    return exp(-dist / p->importance_scale);
}

/* ObservationBuilder.quality_tag */
static eea_real eea_quality_tag(const eea_profile_t *p, eea_real noise_std, bool has_measurement) {
    if (!has_measurement) return 0.0;
    const eea_real ref = p->sensing_noise_std[1];
    if (ref <= 0) return 1.0;
    const eea_real q = 1.0 - 0.8 * noise_std / ref;
    return q < 0.0 ? 0.0 : (q > 1.0 ? 1.0 : q);
}

/* ObservationBuilder.build */
void eea_observation_build(const eea_profile_t *p, const eea_node_state_t *s, float obs[EEA_OBS_DIM]) {
    const eea_real cap = eea_max(s->capacity_j, 1e-9);
    const eea_real age = p->age_scale_s;
    const eea_real href = p->harvest_ref_power_w;
    const eea_real phase = 2.0 * EEA_PI * s->time_of_day_s / EEA_DAY_S;
    eea_real v[EEA_OBS_DIM];
    v[EEA_OBS_BATTERY_SOC] = s->stored_energy_j / cap;
    v[EEA_OBS_HARVEST_POWER] = s->harvest_power_w / href;
    v[EEA_OBS_HARVEST_RECENT] = s->harvest_power_recent_w / href;
    v[EEA_OBS_TOD_SIN] = (sin(phase) + 1.0) / 2.0;
    v[EEA_OBS_TOD_COS] = (cos(phase) + 1.0) / 2.0;
    v[EEA_OBS_MEASUREMENT] = s->has_measurement ? s->measurement_value : 0.0;
    v[EEA_OBS_MEASUREMENT_QUALITY] = eea_quality_tag(p, s->measurement_noise_std, s->has_measurement);
    v[EEA_OBS_MEASUREMENT_AGE] = s->has_measurement ? s->measurement_age_s / age : 1.0;
    v[EEA_OBS_TIME_SINCE_TX_SUCCESS] = s->has_reported ? s->time_since_tx_success_s / age : 1.0;
    v[EEA_OBS_APP_INFO_AGE] = s->has_reported ? s->app_info_age_s / age : 1.0;
    v[EEA_OBS_REPORTED_VALUE] = s->has_reported ? s->reported_value : 0.0;
    v[EEA_OBS_APP_PRIORITY] = (eea_real)s->app_priority / (EEA_N_PRIORITY_LEVELS - 1);
    v[EEA_OBS_IMPORTANCE] = eea_importance(p, s->measurement_value, s->has_measurement);
    v[EEA_OBS_LINK_QUALITY] = s->link_quality;
    v[EEA_OBS_PATH_LOSS_EST] = s->has_link_estimate
                                   ? (s->path_loss_est_db - p->path_loss_min_db) / (p->path_loss_max_db - p->path_loss_min_db)
                                   : 1.0;
    v[EEA_OBS_SENSE_LOW_COST] = s->sensing_energy_low_j / cap;
    v[EEA_OBS_SENSE_HIGH_COST] = s->sensing_energy_high_j / cap;
    v[EEA_OBS_TX_COST] = s->tx_energy_j / cap;
    for (int i = 0; i < EEA_OBS_DIM; ++i) obs[i] = eea_clipf((float)v[i], 0.0f, 1.0f);
}

/* ------------------------------------------------------------------------
 * Energy-feasibility rule
 * ---------------------------------------------------------------------- */
/* actions.plan_execution */
void eea_plan_execution(int sensing_level, int transmit, eea_real stored_energy_j, eea_real baseline_energy_j,
                        eea_real reserve_energy_j, const eea_real *sensing_energy_j, const eea_real *tx_energy_j,
                        int n_modes, bool has_measurement, eea_plan_t *plan) {
    eea_real available = stored_energy_j - baseline_energy_j - reserve_energy_j;
    memset(plan, 0, sizeof(*plan));
    plan->mode = -1;

    int level = sensing_level;
    if (level != 0) {
        const eea_real cost = sensing_energy_j[level];
        if (cost <= available) {
            plan->sensing_energy_j = cost;
            available -= cost;
        } else {
            plan->rejected_sensing_energy = true;
            level = 0;
        }
    }
    plan->sensing_level = level;

    if (transmit > 0 && transmit <= n_modes) {
        const int mode = transmit - 1;
        const eea_real cost = tx_energy_j[mode];
        if (!has_measurement && level == 0) {
            plan->rejected_tx_no_measurement = true;
        } else if (cost <= available) {
            plan->transmit = true;
            plan->tx_energy_j = cost;
            plan->mode = mode;
        } else {
            plan->rejected_tx_energy = true;
        }
    }
}

/* convenience: baseline from the profile, reserve from the gauge capacity */
void eea_plan_execution_profile(const eea_profile_t *p, int sensing_level, int transmit, eea_real stored_energy_j,
                                eea_real capacity_j, bool has_measurement, eea_plan_t *plan) {
    eea_plan_execution(sensing_level, transmit, stored_energy_j, p->baseline_power_w * p->timestep_s,
                       p->reserve_soc * capacity_j, p->sensing_energy_j, p->tx_energy_j, p->n_modes, has_measurement, plan);
}

/* ------------------------------------------------------------------------
 * Rule-based policy
 * ---------------------------------------------------------------------- */
/* RuleBasedPolicy._tx : transmit action value = 1 + chosen radio mode */
int eea_rule_based_tx(const eea_rule_params_t *rp, const eea_profile_t *p, const float obs[EEA_OBS_DIM]) {
    if (p->n_modes == 1) return 1 + p->reference_mode;
    const eea_real pl_norm = (eea_real)obs[EEA_OBS_PATH_LOSS_EST];
    if (pl_norm >= 1.0) return 1 + p->reference_mode; /* no estimate yet */
    const eea_real pl_db = p->path_loss_min_db + pl_norm * (p->path_loss_max_db - p->path_loss_min_db);
    /* modes sorted by energy, cheapest first (stable, like Python's sorted) */
    int order[EEA_MAX_MODES];
    for (int k = 0; k < p->n_modes; ++k) {
        int j = k;
        while (j > 0 && p->tx_energy_j[order[j - 1]] > p->tx_energy_j[k]) {
            order[j] = order[j - 1];
            --j;
        }
        order[j] = k;
    }
    int pos = p->n_modes - 1;
    for (int i = 0; i < p->n_modes; ++i) {
        if (eea_margin_for_mode(p, order[i], pl_db) >= rp->link_margin_target_db) {
            pos = i;
            break;
        }
    }
    if ((eea_real)obs[EEA_OBS_LINK_QUALITY] < rp->link_quality_escalate) {
        pos = pos + 1 < p->n_modes ? pos + 1 : p->n_modes - 1;
    }
    return 1 + order[pos];
}

/* RuleBasedPolicy.act */
void eea_rule_based_act(const eea_rule_params_t *rp, const eea_profile_t *p, const float obs[EEA_OBS_DIM],
                        int *sensing_level, int *transmit) {
    const eea_real soc = (eea_real)obs[EEA_OBS_BATTERY_SOC];
    const eea_real harvest_recent = (eea_real)obs[EEA_OBS_HARVEST_RECENT];
    const eea_real meas = (eea_real)obs[EEA_OBS_MEASUREMENT];
    const eea_real quality = (eea_real)obs[EEA_OBS_MEASUREMENT_QUALITY];
    const bool has_measurement = quality > 0.0;
    const bool high_quality = quality > 0.5;
    const eea_real meas_age_h = (eea_real)obs[EEA_OBS_MEASUREMENT_AGE] * rp->age_scale_h;
    const eea_real since_ack_h = (eea_real)obs[EEA_OBS_TIME_SINCE_TX_SUCCESS] * rp->age_scale_h;
    const eea_real app_age_h = (eea_real)obs[EEA_OBS_APP_INFO_AGE] * rp->age_scale_h;
    const eea_real reported = (eea_real)obs[EEA_OBS_REPORTED_VALUE];
    const bool has_reported = (eea_real)obs[EEA_OBS_TIME_SINCE_TX_SUCCESS] < 1.0;
    int priority = (int)lround((eea_real)obs[EEA_OBS_APP_PRIORITY] * 2.0); /* 0 / 1 / 2 */
    if (priority < 0) priority = 0;
    if (priority > 2) priority = 2;
    const eea_real importance = (eea_real)obs[EEA_OBS_IMPORTANCE];
    const bool urgent = priority >= 2;
    const bool eco = soc < rp->soc_low && !urgent;
    const bool generous = (soc > rp->soc_high || harvest_recent > rp->harvest_strong) && !eco;
    const bool unreported = has_measurement && (since_ack_h > meas_age_h + 1e-6);
    const eea_real delta = (has_measurement && has_reported) ? fabs(meas - reported) : (has_measurement ? 1.0 : 0.0);

    eea_real interval_h = rp->report_interval_h[priority];
    if (eco) interval_h *= rp->report_interval_eco_factor;
    else if (generous && priority == 0) interval_h *= rp->report_interval_generous_factor;

    /* 1. deep economy */
    if (soc < rp->soc_critical) {
        const eea_real limit_h = urgent ? rp->report_interval_h[2] : rp->deep_eco_interval_h;
        if (app_age_h >= limit_h) { *sensing_level = 2; *transmit = eea_rule_based_tx(rp, p, obs); return; }
        *sensing_level = 0; *transmit = 0; return;
    }
    /* 2. retry a fresh, unacknowledged high-quality report (link permitting) */
    const bool link_ok = (eea_real)obs[EEA_OBS_LINK_QUALITY] >= rp->retry_min_link_quality;
    if (unreported && high_quality && link_ok && meas_age_h < rp->retry_age_h && app_age_h > interval_h) {
        *sensing_level = 0; *transmit = eea_rule_based_tx(rp, p, obs); return;
    }
    /* 3. scheduled report */
    if (app_age_h >= interval_h) {
        *sensing_level = eco ? rp->eco_sensing_level : 2; *transmit = eea_rule_based_tx(rp, p, obs); return;
    }
    /* 4. event / importance report */
    if (unreported && (delta > rp->event_delta || (importance > rp->importance_immediate && delta > rp->importance_delta))) {
        *sensing_level = 2; *transmit = eea_rule_based_tx(rp, p, obs); return;
    }
    /* 5. cheap check between reports */
    if (!eco && (!has_measurement || meas_age_h >= rp->check_interval_h)) { *sensing_level = 1; *transmit = 0; return; }
    *sensing_level = 0; *transmit = 0;
}

/* ------------------------------------------------------------------------
 * MLP policy (float32, like rl.NumpyMLPPolicy)
 * ---------------------------------------------------------------------- */
static int eea_argmax(const float *x, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) if (x[i] > x[best]) best = i; /* first maximum, like np.argmax */
    return best;
}

/* NumpyMLPPolicy.forward */
void eea_mlp_forward(const eea_mlp_t *m, const float obs[EEA_OBS_DIM], float *out) {
    float buf[2][EEA_MAX_UNITS];
    const float *h = obs;
    int cur = 0;
    for (int l = 0; l < m->n_layers; ++l) {
        const eea_layer_t *L = &m->layers[l];
        float *y = (l == m->n_layers - 1) ? out : buf[cur];
        for (int i = 0; i < L->out_dim; ++i) {
            float acc = 0.0f;
            const float *w = L->W + (size_t)i * L->in_dim;
            for (int j = 0; j < L->in_dim; ++j) acc += w[j] * h[j];
            acc += L->b[i];
            if (L->activation == EEA_ACT_TANH) acc = tanhf(acc);
            else if (L->activation == EEA_ACT_RELU) acc = acc > 0.0f ? acc : 0.0f;
            y[i] = acc;
        }
        h = y;
        cur ^= 1;
    }
}

/* NumpyMLPPolicy.act */
void eea_mlp_act(const eea_mlp_t *m, const float obs[EEA_OBS_DIM], int *sensing_level, int *transmit) {
    float out[EEA_MAX_UNITS];
    eea_mlp_forward(m, obs, out);
    const int n_out = m->layers[m->n_layers - 1].out_dim;
    if (m->output_kind == EEA_OUT_FLAT_Q_VALUES) {
        const int idx = eea_argmax(out, n_out);
        const int n_tx = 1 + m->n_modes;
        *sensing_level = idx / n_tx;
        *transmit = idx % n_tx;
    } else {
        const int n1 = m->output_split[0];
        *sensing_level = eea_argmax(out, n1);
        *transmit = eea_argmax(out + n1, n_out - n1);
    }
}
