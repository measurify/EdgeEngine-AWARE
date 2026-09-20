/*
 * Example decision loop for an EdgeEngine AWARE node (documentation build).
 *
 * This file shows how the runtime (eea_node.c) and the generated policy data
 * (eea_policy_data.h, from tools/export_c.py) are wired to a board. The HAL
 * functions at the top are stubs: replace them with your drivers (fuel gauge,
 * harvester monitor, soil-moisture sensor, LoRa stack, RTC / low-power timer).
 * It mirrors deployment.NodeController in Python step by step.
 *
 * Build (host, with a rule-based or MLP header generated into this folder):
 *   python tools/export_c.py examples/bundles/ppo_default.json -o firmware/eea_policy_data.h
 *   cc -std=c99 -O2 -I firmware firmware/eea_node.c firmware/main_example.c -lm -o node_example
 */
#include <stdio.h>

#include "eea_node.h"
#include "eea_policy_data.h"

/* ------------------------------------------------------------------ HAL --- */
static double hal_rtc_now_s(void) { static double t = 0.0; return t += EEA_PROFILE.timestep_s; }
static double hal_rtc_time_of_day_s(double now_s) { return now_s - 86400.0 * (double)(long)(now_s / 86400.0); }
static double hal_gauge_energy_j(void) { return 180.0; }              /* fuel gauge or V -> E lookup */
static double hal_gauge_capacity_j(void) { return 300.0; }
static double hal_harvester_avg_power_w(void) { return 0.0012; }      /* energy since last wake-up / dt */
static int hal_downlink_priority(void) { return -1; }                /* -1: no new downlink */
static double hal_sensor_read(int level) { (void)level; return 0.42; }
/* returns 1 = ACK, 0 = no ACK, -1 = unconfirmed link; *margin_db from the ACK SNR/LinkCheck */
static int hal_lora_send(double value, int mode, bool *has_margin, double *margin_db) {
    (void)value; (void)mode; *has_margin = true; *margin_db = 6.0; return 1;
}
static void hal_sleep_until_next_slot(void) {}

/* ------------------------------------------------------------- policy --- */
static void policy_act(const float obs[EEA_OBS_DIM], int *sense, int *tx) {
#if defined(EEA_HAS_MLP)
    eea_mlp_act(&EEA_MLP, obs, sense, tx);
#elif defined(EEA_HAS_RULE_PARAMS)
    eea_rule_based_act(&EEA_RULE_PARAMS, &EEA_PROFILE, obs, sense, tx);
#else
#error "the policy header defines neither EEA_MLP nor EEA_RULE_PARAMS"
#endif
}

/* ---------------------------------------------------------- main loop --- */
int main(void) {
    static eea_tracker_t tracker;   /* persists across wake-ups (keep in retained RAM) */
    eea_tracker_init(&tracker, &EEA_PROFILE);

    for (int cycle = 0; cycle < 8; ++cycle) {
        /* 1. sample the measurable state */
        const double now = hal_rtc_now_s();
        eea_tracker_begin_step(&tracker, now, hal_rtc_time_of_day_s(now), hal_gauge_energy_j(), hal_gauge_capacity_j(),
                               hal_harvester_avg_power_w(), hal_downlink_priority());

        /* 2. observation == ObservationBuilder.build */
        eea_node_state_t state;
        float obs[EEA_OBS_DIM];
        eea_tracker_state(&tracker, &state);
        eea_observation_build(&EEA_PROFILE, &state, obs);

        /* 3. decision */
        int sense = 0, tx = 0;
        policy_act(obs, &sense, &tx);

        /* 4. feasibility == actions.plan_execution */
        eea_plan_t plan;
        eea_plan_execution_profile(&EEA_PROFILE, sense, tx, state.stored_energy_j, state.capacity_j, state.has_measurement, &plan);

        /* 5. execute */
        if (plan.sensing_level != 0) {
            eea_measurement_t m = {hal_sensor_read(plan.sensing_level), now, plan.sensing_level,
                                   EEA_PROFILE.sensing_noise_std[plan.sensing_level]};
            eea_tracker_on_measurement(&tracker, &m);
        }
        if (plan.transmit) {
            bool has_margin;
            double margin;
            const int acked = hal_lora_send(tracker.measurement.value, plan.mode, &has_margin, &margin);
            eea_tracker_on_transmission(&tracker, now, acked, now, plan.mode, has_margin, margin);
        }

        printf("t=%7.0f soc=%.2f obs[age]=%.3f -> requested (%d,%d) executed sense=%d tx=%d mode=%d%s\n", now,
               (double)obs[EEA_OBS_BATTERY_SOC], (double)obs[EEA_OBS_APP_INFO_AGE], sense, tx, plan.sensing_level,
               plan.transmit ? 1 : 0, plan.mode, plan.rejected_tx_energy ? " (tx rejected: energy)" : "");
        hal_sleep_until_next_slot();
    }
    return 0;
}
