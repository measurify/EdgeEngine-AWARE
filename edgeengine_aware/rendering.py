"""Matplotlib dashboard renderer.

One figure is created lazily and updated in place at every ``render()`` call.
``human`` shows the figure interactively (no-op on a non-interactive backend,
so headless training scripts are never blocked); ``rgb_array`` draws on an
Agg canvas and returns an ``(H, W, 3)`` uint8 array; ``ansi`` returns a
compact text dashboard. Only privileged information that the *user* is
allowed to see is plotted (true moisture, true harvest); nothing here feeds
the policy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .env import EdgeEngineAwareEnv

PRIORITY_NAMES = ("routine", "elevated", "URGENT")
SENSE_NAMES = ("-", "low", "HIGH")


def text_dashboard(env: "EdgeEngineAwareEnv") -> str:
    """Compact one-screen textual status of the environment."""
    gt = env.ground_truth()
    ns = env.node_state()
    last = env._last_step
    tx = "-" if not last["tx_attempted"] else ("ok" if last["tx_success"] else "FAIL")
    meas = f"{ns.measurement_value:.3f}" if ns.has_measurement else "  n/a"
    app = f"{gt['app_last_value']:.3f}" if gt["app_last_value"] is not None else "  n/a"
    day, hour = env.clock.day_index(), env.clock.hour_of_day()
    lines = [
        f"EdgeEngine AWARE | step {env._step_count:4d} | day {day} {int(hour):02d}:{int((hour % 1) * 60):02d}",
        f"  battery SoC        : {ns.soc():6.1%}   ({ns.stored_energy_j:7.1f} J / {ns.capacity_j:.0f} J)",
        f"  harvest (meas/true): {ns.harvest_power_w*1e3:6.2f} / {gt['harvest_power_true_w']*1e3:6.2f} mW   recent {ns.harvest_power_recent_w*1e3:.2f} mW   last step {env._last_harvested_j:.2f} J",
        f"  soil moisture      : true {gt['soil_moisture']:.3f} | node {meas} (age {ns.measurement_age_s/3600:.1f} h) | app {app} (AoI {gt['app_aoi_s']/3600:.1f} h)",
        f"  zone / priority    : {('normal','warning','CRITICAL')[gt['zone']]} / {PRIORITY_NAMES[gt['app_priority']]}",
        f"  action             : sense={SENSE_NAMES[last['sensing_level']]}  tx={tx}",
        f"  reward             : {last['reward']:+.3f}   (utility {last['utility']:.3f})",
    ]
    return "\n".join(lines)


class DashboardRenderer:
    def __init__(self, env: "EdgeEngineAwareEnv"):
        self.env = env
        self.fig = None
        self._interactive = False
        self._pyplot_managed = False

    # ------------------------------------------------------------------
    def _ensure_figure(self, mode: str):
        import matplotlib

        if self.fig is not None:
            return
        if mode == "rgb_array":
            # Do not touch the global backend; draw on a private Agg canvas.
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure

            self.fig = Figure(figsize=(12, 10), dpi=80)
            FigureCanvasAgg(self.fig)
        else:
            import matplotlib.pyplot as plt

            self._interactive = matplotlib.get_backend().lower() not in ("agg", "pdf", "svg", "ps", "template")
            if self._interactive:
                plt.ion()
            self.fig = plt.figure(figsize=(12, 10), dpi=80)
            self._pyplot_managed = True
        self.axes = self.fig.subplots(4, 2, sharex=True)
        self.fig.subplots_adjust(hspace=0.4, wspace=0.25, top=0.84, bottom=0.06)
        self._title = self.fig.text(0.02, 0.985, "", fontsize=9, family="monospace", ha="left", va="top")

    def _draw(self) -> None:
        env = self.env
        log = env.log
        cfg = env.cfg
        t = np.asarray(log.time_s) / 86400.0 if len(log) else np.zeros(0)
        axes = self.axes
        for ax in axes.ravel():
            ax.cla()
            ax.grid(alpha=0.3)

        def step_series(values):
            return np.asarray(values, dtype=float)

        # battery
        ax = axes[0, 0]
        ax.plot(t, step_series(log.soc), color="tab:blue")
        ax.axhline(cfg.reward.safe_soc, color="tab:red", ls="--", lw=0.8)
        ax.set_ylim(0, 1.02)
        ax.set_title("Battery SoC")

        # harvest
        ax = axes[0, 1]
        ax.plot(t, step_series(log.harvest_power_w) * 1e3, color="tab:orange", lw=0.9)
        ax.set_title("Measured harvesting power [mW]")

        # moisture
        ax = axes[1, 0]
        ax.plot(t, step_series(log.true_moisture), color="k", lw=1.0, label="true")
        ax.plot(t, step_series(log.measured_moisture), color="tab:green", lw=0.9, label="node", drawstyle="steps-post")
        ax.plot(t, step_series(log.app_moisture), color="tab:purple", lw=0.9, label="application", drawstyle="steps-post")
        ax.axhline(cfg.agriculture.warning_threshold, color="tab:orange", ls=":", lw=0.8)
        ax.axhline(cfg.agriculture.critical_threshold, color="tab:red", ls=":", lw=0.8)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", fontsize=7, ncol=3)
        ax.set_title("Soil moisture (true / node / application)")

        # events
        ax = axes[1, 1]
        if len(log):
            lvl = step_series(log.sensing_level)
            att = step_series(log.tx_attempt)
            suc = step_series(log.tx_success)
            ax.vlines(t[lvl == 1], 0, 0.8, color="tab:green", alpha=0.5, lw=0.8, label="sense low")
            ax.vlines(t[lvl == 2], 0, 1.0, color="darkgreen", alpha=0.8, lw=0.8, label="sense high")
            ax.scatter(t[(att == 1) & (suc == 1)], np.full(int(((att == 1) & (suc == 1)).sum()), 1.3), marker="^", s=12, color="tab:purple", label="tx ok")
            ax.scatter(t[(att == 1) & (suc == 0)], np.full(int(((att == 1) & (suc == 0)).sum()), 1.3), marker="x", s=14, color="tab:red", label="tx fail")
            ax.legend(loc="upper right", fontsize=7, ncol=4)
        ax.set_ylim(0, 1.6)
        ax.set_yticks([])
        ax.set_title("Sensing and transmission events")

        # AoI
        ax = axes[2, 0]
        ax.plot(t, step_series(log.aoi_s) / 3600.0, color="tab:purple")
        ax.set_title("Age of information at the application [h]")

        # priority
        ax = axes[2, 1]
        ax.step(t, step_series(log.priority), where="post", color="tab:red")
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(PRIORITY_NAMES, fontsize=7)
        ax.set_ylim(-0.2, 2.2)
        ax.set_title("Application priority")

        # reward
        ax = axes[3, 0]
        ax.plot(t, step_series(log.reward), color="tab:gray", lw=0.7, label="reward")
        ax.plot(t, step_series(log.utility), color="tab:blue", lw=0.9, label="utility")
        ax.legend(loc="upper right", fontsize=7)
        ax.set_title("Instantaneous reward")
        ax.set_xlabel("time [days]")

        ax = axes[3, 1]
        if len(log):
            ax.plot(t, np.cumsum(step_series(log.reward)), color="tab:gray")
        ax.set_title("Cumulative reward")
        ax.set_xlabel("time [days]")

        self._title.set_text(text_dashboard(env))

    # ------------------------------------------------------------------
    def render(self, mode: str):
        if mode == "ansi":
            return text_dashboard(self.env)
        self._ensure_figure(mode)
        self._draw()
        if mode == "rgb_array":
            self.fig.canvas.draw()
            buf = np.asarray(self.fig.canvas.buffer_rgba())
            return buf[..., :3].copy()
        # human
        self.fig.canvas.draw_idle()
        if self._interactive:
            import matplotlib.pyplot as plt

            plt.pause(1.0 / self.env.metadata["render_fps"])
        return None

    def close(self) -> None:
        if self.fig is not None and self._pyplot_managed:
            import matplotlib.pyplot as plt

            plt.close(self.fig)
        self.fig = None
