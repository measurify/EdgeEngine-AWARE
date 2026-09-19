/*
 * Test harness: drives the C runtime from stdin, one command per line, and
 * prints results in a form Python can parse exactly (floats as C99 hex
 * literals). Used by tests/test_firmware.py; not part of the firmware.
 *
 * Commands
 *   Z                                   reset the tracker
 *   B now tod energy cap harvest prio   tracker_begin_step (prio < 0: no downlink)
 *   M value ts level noise              tracker_on_measurement
 *   X sent_at acked now mode has_margin margin
 *                                       tracker_on_transmission (acked -1/0/1)
 *   Q prio                              tracker_set_priority
 *   O                                   print the 18 observation floats (hex)
 *   R                                   rule-based action on the current state: "sense tx"
 *   N                                   MLP action on the current state: "sense tx" + logits (hex)
 *   V f0 .. f17                         set an explicit observation (hex floats) for R/N
 *   P sense tx stored cap has_meas      plan_execution (profile form)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "eea_node.h"
#include "eea_policy_data.h"

static void print_obs(const float *o, int n) {
    for (int i = 0; i < n; ++i) printf("%s%a", i ? " " : "", (double)o[i]);
    printf("\n");
}

int main(void) {
    eea_tracker_t tracker;
    eea_tracker_init(&tracker, &EEA_PROFILE);
    eea_node_state_t state;
    float obs[EEA_OBS_DIM] = {0};
    bool explicit_obs = false;
    char line[8192];

    while (fgets(line, sizeof line, stdin)) {
        char cmd = line[0];
        const char *args = line + 1;
        if (cmd == 'Z') {
            eea_tracker_reset(&tracker);
            explicit_obs = false;
            printf("ok\n");
        } else if (cmd == 'B') {
            double now, tod, energy, cap, harvest;
            int prio;
            if (sscanf(args, "%lf %lf %lf %lf %lf %d", &now, &tod, &energy, &cap, &harvest, &prio) != 6) return 2;
            eea_tracker_begin_step(&tracker, now, tod, energy, cap, harvest, prio);
            explicit_obs = false;
            printf("ok\n");
        } else if (cmd == 'M') {
            eea_measurement_t m;
            if (sscanf(args, "%lf %lf %d %lf", &m.value, &m.timestamp_s, &m.level, &m.noise_std) != 4) return 2;
            eea_tracker_on_measurement(&tracker, &m);
            explicit_obs = false;
            printf("ok\n");
        } else if (cmd == 'X') {
            double sent_at, now, margin;
            int acked, mode, has_margin;
            if (sscanf(args, "%lf %d %lf %d %d %lf", &sent_at, &acked, &now, &mode, &has_margin, &margin) != 6) return 2;
            eea_tracker_on_transmission(&tracker, sent_at, acked, now, mode, has_margin != 0, margin);
            explicit_obs = false;
            printf("ok\n");
        } else if (cmd == 'Q') {
            int prio;
            if (sscanf(args, "%d", &prio) != 1) return 2;
            eea_tracker_set_priority(&tracker, prio);
            printf("ok\n");
        } else if (cmd == 'O') {
            eea_tracker_state(&tracker, &state);
            eea_observation_build(&EEA_PROFILE, &state, obs);
            explicit_obs = false;
            print_obs(obs, EEA_OBS_DIM);
        } else if (cmd == 'V') {
            const char *p = args;
            for (int i = 0; i < EEA_OBS_DIM; ++i) {
                char *end;
                obs[i] = (float)strtod(p, &end);
                if (end == p) return 2;
                p = end;
            }
            explicit_obs = true;
            printf("ok\n");
        } else if (cmd == 'R' || cmd == 'N') {
            if (!explicit_obs) {
                eea_tracker_state(&tracker, &state);
                eea_observation_build(&EEA_PROFILE, &state, obs);
            }
            int sense = 0, tx = 0;
            if (cmd == 'R') {
#ifdef EEA_HAS_RULE_PARAMS
                eea_rule_based_act(&EEA_RULE_PARAMS, &EEA_PROFILE, obs, &sense, &tx);
                printf("%d %d\n", sense, tx);
#else
                printf("unavailable\n");
#endif
            } else {
#ifdef EEA_HAS_MLP
                float out[EEA_MAX_UNITS];
                eea_mlp_forward(&EEA_MLP, obs, out);
                eea_mlp_act(&EEA_MLP, obs, &sense, &tx);
                printf("%d %d", sense, tx);
                const int n_out = EEA_MLP.layers[EEA_MLP.n_layers - 1].out_dim;
                for (int i = 0; i < n_out; ++i) printf(" %a", (double)out[i]);
                printf("\n");
#else
                printf("unavailable\n");
#endif
            }
        } else if (cmd == 'P') {
            int sense, tx, has_meas;
            double stored, cap;
            if (sscanf(args, "%d %d %lf %lf %d", &sense, &tx, &stored, &cap, &has_meas) != 5) return 2;
            eea_plan_t plan;
            eea_plan_execution_profile(&EEA_PROFILE, sense, tx, stored, cap, has_meas != 0, &plan);
            printf("%d %d %d %a %a %d %d %d\n", plan.sensing_level, plan.transmit ? 1 : 0, plan.mode,
                   (double)plan.sensing_energy_j, (double)plan.tx_energy_j, plan.rejected_sensing_energy ? 1 : 0,
                   plan.rejected_tx_no_measurement ? 1 : 0, plan.rejected_tx_energy ? 1 : 0);
        } else if (cmd == '\n' || cmd == '#') {
            continue;
        } else {
            fprintf(stderr, "unknown command %c\n", cmd);
            return 2;
        }
        fflush(stdout);
    }
    return 0;
}
