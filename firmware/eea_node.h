/*
 * EdgeEngine AWARE - node-side runtime in portable C99.
 *
 * Line-by-line port of the Python modules that define the sim-to-real
 * contract:
 *
 *   observation.py  NodeProfile / NodeStateTracker / ObservationBuilder
 *   actions.py      plan_execution (energy-feasibility rule)
 *   policies.py     RuleBasedPolicy
 *   rl.py           NumpyMLPPolicy (small dense network, argmax decoding)
 *
 * The equivalence with the Python reference is checked by
 * tests/test_firmware.py, which compiles firmware/test/eea_harness.c with the
 * generated policy data (tools/export_c.py) and compares observations and
 * actions event by event.
 *
 * Conventions
 * -----------
 * * All physical quantities are double, exactly like the Python reference
 *   (the observation is rounded to float32 at the very end, as numpy does).
 *   On a single-precision target you may define EEA_REAL as float; the
 *   observation may then differ from Python in the last bit (that build is
 *   not covered by the equivalence tests).
 * * No dynamic allocation, no I/O, no dependency beyond <math.h>.
 * * Radio modes are indexed 0..n_modes-1; the "transmit" action value is
 *   0 = no transmission, 1 + mode = transmit with that mode.
 */
#ifndef EEA_NODE_H
#define EEA_NODE_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef EEA_REAL
#define EEA_REAL double
#endif
typedef EEA_REAL eea_real;

#define EEA_OBS_DIM 18
#define EEA_N_SENSING_LEVELS 3
#define EEA_MAX_MODES 4
#define EEA_MAX_LAYERS 4
#define EEA_MAX_UNITS 128

/* observation indices (order of observation.OBSERVATION_FIELDS) */
enum {
    EEA_OBS_BATTERY_SOC = 0,
    EEA_OBS_HARVEST_POWER,
    EEA_OBS_HARVEST_RECENT,
    EEA_OBS_TOD_SIN,
    EEA_OBS_TOD_COS,
    EEA_OBS_MEASUREMENT,
    EEA_OBS_MEASUREMENT_QUALITY,
    EEA_OBS_MEASUREMENT_AGE,
    EEA_OBS_TIME_SINCE_TX_SUCCESS,
    EEA_OBS_APP_INFO_AGE,
    EEA_OBS_REPORTED_VALUE,
    EEA_OBS_APP_PRIORITY,
    EEA_OBS_IMPORTANCE,
    EEA_OBS_LINK_QUALITY,
    EEA_OBS_PATH_LOSS_EST,
    EEA_OBS_SENSE_LOW_COST,
    EEA_OBS_SENSE_HIGH_COST,
    EEA_OBS_TX_COST
};

/* ------------------------------------------------------------------------
 * Flash constants (observation.NodeProfile + ObservationConfig)
 * ---------------------------------------------------------------------- */
typedef struct {
    eea_real sensing_energy_j[EEA_N_SENSING_LEVELS];
    eea_real sensing_noise_std[EEA_N_SENSING_LEVELS];
    int n_modes;
    eea_real tx_energy_j[EEA_MAX_MODES];
    eea_real tx_power_dbm[EEA_MAX_MODES];
    eea_real sensitivity_dbm[EEA_MAX_MODES];
    int reference_mode;
    eea_real warning_threshold;
    eea_real critical_threshold;
    eea_real timestep_s;
    eea_real baseline_power_w;
    eea_real reserve_soc;
    bool ack_available;
    bool critical_is_upper;   /* danger side of the monitored quantity: false = low (soil moisture), true = high (CO2, temperature) */
    /* ObservationConfig */
    eea_real harvest_ref_power_w;
    eea_real age_scale_s;
    eea_real harvest_ewma_alpha;
    eea_real link_ewma_alpha;
    eea_real importance_scale;
    eea_real path_loss_min_db;
    eea_real path_loss_max_db;
} eea_profile_t;

/* ------------------------------------------------------------------------
 * Node-side tracker (observation.NodeStateTracker)
 * ---------------------------------------------------------------------- */
typedef struct {
    eea_real value;
    eea_real timestamp_s;
    int level;
    eea_real noise_std;
} eea_measurement_t;

typedef struct {
    const eea_profile_t *profile;
    eea_real now_s;
    eea_real time_of_day_s;
    eea_real energy_j;
    eea_real capacity_j;
    eea_real harvest_w;
    eea_real harvest_recent_w;
    bool harvest_initialised;
    bool has_measurement;
    eea_measurement_t measurement;
    bool has_ack;
    eea_real last_ack_time_s;
    eea_measurement_t last_ack_measurement;
    eea_real last_ack_sent_at_s;
    int priority;
    eea_real link_quality;
    bool has_path_loss_est;
    eea_real path_loss_est_db;
} eea_tracker_t;

/* Hardware-measurable state (observation.NodeState) */
typedef struct {
    eea_real time_of_day_s;
    eea_real stored_energy_j;
    eea_real capacity_j;
    eea_real harvest_power_w;
    eea_real harvest_power_recent_w;
    bool has_measurement;
    eea_real measurement_value;
    eea_real measurement_noise_std;
    eea_real measurement_age_s;
    bool has_reported;
    eea_real time_since_tx_success_s;
    eea_real app_info_age_s;
    eea_real reported_value;
    int app_priority;
    eea_real link_quality;
    bool has_link_estimate;
    eea_real path_loss_est_db;
    eea_real sensing_energy_low_j;
    eea_real sensing_energy_high_j;
    eea_real tx_energy_j;
} eea_node_state_t;

void eea_tracker_init(eea_tracker_t *t, const eea_profile_t *profile);
void eea_tracker_reset(eea_tracker_t *t);
/* priority < 0 means "no new downlink" (keep the previous one) */
void eea_tracker_begin_step(eea_tracker_t *t, eea_real now_s, eea_real time_of_day_s, eea_real energy_j,
                            eea_real capacity_j, eea_real harvest_power_w, int priority);
void eea_tracker_on_measurement(eea_tracker_t *t, const eea_measurement_t *m);
/* acked: 1 = ACK received, 0 = no ACK, -1 = link without confirmations.
 * has_margin/margin_db: link margin measured from the ACK (if any). */
void eea_tracker_on_transmission(eea_tracker_t *t, eea_real sent_at_s, int acked, eea_real now_s, int mode,
                                 bool has_margin, eea_real margin_db);
void eea_tracker_set_priority(eea_tracker_t *t, int priority);
void eea_tracker_state(const eea_tracker_t *t, eea_node_state_t *s);

/* ------------------------------------------------------------------------
 * Observation (observation.ObservationBuilder.build) - float32 output
 * ---------------------------------------------------------------------- */
void eea_observation_build(const eea_profile_t *p, const eea_node_state_t *s, float obs[EEA_OBS_DIM]);

/* ------------------------------------------------------------------------
 * Energy-feasibility rule (actions.plan_execution)
 * ---------------------------------------------------------------------- */
typedef struct {
    int sensing_level;   /* executed level, 0 if rejected/not requested */
    bool transmit;
    int mode;            /* -1 when no transmission */
    eea_real sensing_energy_j;
    eea_real tx_energy_j;
    bool rejected_sensing_energy;
    bool rejected_tx_no_measurement;
    bool rejected_tx_energy;
} eea_plan_t;

/* transmit: 0 = none, 1 + mode = transmit with that mode */
void eea_plan_execution(int sensing_level, int transmit, eea_real stored_energy_j, eea_real baseline_energy_j,
                        eea_real reserve_energy_j, const eea_real *sensing_energy_j, const eea_real *tx_energy_j,
                        int n_modes, bool has_measurement, eea_plan_t *plan);
/* baseline = profile.baseline_power_w * timestep_s, reserve = profile.reserve_soc * capacity_j */
void eea_plan_execution_profile(const eea_profile_t *p, int sensing_level, int transmit, eea_real stored_energy_j,
                                eea_real capacity_j, bool has_measurement, eea_plan_t *plan);

/* ------------------------------------------------------------------------
 * Rule-based policy (policies.RuleBasedPolicy)
 * ---------------------------------------------------------------------- */
typedef struct {
    eea_real soc_critical;
    eea_real soc_low;
    eea_real deep_eco_interval_h;
    eea_real soc_high;
    eea_real harvest_strong;
    eea_real report_interval_h[3];
    eea_real report_interval_eco_factor;
    int eco_sensing_level;
    eea_real report_interval_generous_factor;
    eea_real check_interval_h;
    eea_real event_delta;
    eea_real importance_immediate;
    eea_real importance_delta;
    eea_real retry_age_h;
    eea_real age_scale_h;
    eea_real link_margin_target_db;
    eea_real link_quality_escalate;
    eea_real retry_min_link_quality;
} eea_rule_params_t;

void eea_rule_based_act(const eea_rule_params_t *rp, const eea_profile_t *p, const float obs[EEA_OBS_DIM],
                        int *sensing_level, int *transmit);
int eea_rule_based_tx(const eea_rule_params_t *rp, const eea_profile_t *p, const float obs[EEA_OBS_DIM]);

/* ------------------------------------------------------------------------
 * Small MLP policy (rl.NumpyMLPPolicy) - float32 arithmetic
 * ---------------------------------------------------------------------- */
enum { EEA_ACT_LINEAR = 0, EEA_ACT_TANH = 1, EEA_ACT_RELU = 2 };
enum { EEA_OUT_MULTIDISCRETE_LOGITS = 0, EEA_OUT_FLAT_Q_VALUES = 1 };

typedef struct {
    int in_dim;
    int out_dim;
    const float *W;   /* row-major [out_dim][in_dim] */
    const float *b;   /* [out_dim] */
    int activation;
} eea_layer_t;

typedef struct {
    int n_layers;
    eea_layer_t layers[EEA_MAX_LAYERS];
    int output_kind;
    int output_split[2];   /* multidiscrete: [3, 1 + n_modes] */
    int n_modes;
} eea_mlp_t;

/* forward pass; out must hold layers[n_layers-1].out_dim floats */
void eea_mlp_forward(const eea_mlp_t *m, const float obs[EEA_OBS_DIM], float *out);
void eea_mlp_act(const eea_mlp_t *m, const float obs[EEA_OBS_DIM], int *sensing_level, int *transmit);

/* action encoding helpers (actions.py) */
static inline int eea_flat_action(int sensing_level, int transmit, int n_modes) { return sensing_level * (1 + n_modes) + transmit; }

#ifdef __cplusplus
}
#endif
#endif /* EEA_NODE_H */
