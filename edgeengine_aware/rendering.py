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
_MODE_COLORS = ("#eda100", "#2a78d6", "#4a3aa7", "#1baf7a", "#e87ba4", "#008300")


def plt_mode_color(k: int, n_modes: int) -> str:
    return _MODE_COLORS[k % len(_MODE_COLORS)]


def text_dashboard(env: "EdgeEngineAwareEnv") -> str:
    """Compact one-screen textual status of the environment."""
    gt = env.ground_truth()
    ns = env.node_state()
    last = env._last_step
    if not last["tx_attempted"]:
        tx = "-"
    else:
        mode_name = env.cfg.communication.modes[last["tx_mode"]].name if last["tx_mode"] >= 0 else "?"
        tx = f"{mode_name}:{'ok' if last['tx_success'] else 'FAIL'}"
    meas = f"{ns.measurement_value:.3f}" if ns.has_measurement else "  n/a"
    app = f"{gt['app_last_value']:.3f}" if gt["app_last_value"] is not None else "  n/a"
    day, hour = env.clock.day_index(), env.clock.hour_of_day()
    q = env.quantity
    activity = f"   activity {gt['activity']:.2f}" if gt.get("activity") is not None else ""
    lines = [
        f"EdgeEngine AWARE [{env.cfg.domain}] | step {env._step_count:4d} | day {day} {int(hour):02d}:{int((hour % 1) * 60):02d}{activity}",
        f"  battery SoC        : {ns.soc():6.1%}   ({ns.stored_energy_j:7.1f} J / {ns.capacity_j:.0f} J)",
        f"  harvest (meas/true): {ns.harvest_power_w*1e3:6.3f} / {gt['harvest_power_true_w']*1e3:6.3f} mW   recent {ns.harvest_power_recent_w*1e3:.3f} mW   last step {env._last_harvested_j:.3f} J",
        f"  {q.name[:19]:19s}: true {gt['value']:.3f} ({gt['physical_value']:.0f} {q.unit[:12]}) | node {meas} (age {ns.measurement_age_s/3600:.1f} h) | app {app} (AoI {gt['app_aoi_s']/3600:.1f} h)",
        f"  zone / priority    : {('normal','warning','CRITICAL')[gt['zone']]} / {PRIORITY_NAMES[gt['app_priority']]}   path loss {gt['path_loss_db']:.0f} dB (node est. {ns.path_loss_est_db:.0f} dB)",
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
        ax.set_title(f"Measured harvesting power [mW] ({cfg.harvesting_source})")

        # monitored quantity
        q = env.quantity
        ax = axes[1, 0]
        ax.plot(t, step_series(log.true_moisture), color="k", lw=1.0, label="true")
        ax.plot(t, step_series(log.measured_moisture), color="tab:green", lw=0.9, label="node", drawstyle="steps-post")
        ax.plot(t, step_series(log.app_moisture), color="tab:purple", lw=0.9, label="application", drawstyle="steps-post")
        ax.axhline(q.warning_threshold, color="tab:orange", ls=":", lw=0.8)
        ax.axhline(q.critical_threshold, color="tab:red", ls=":", lw=0.8)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", fontsize=7, ncol=3)
        ax.set_title(f"{q.name} (true / node / application), normalised; danger {'above' if q.critical_is_upper else 'below'} the dotted lines")

        # events
        ax = axes[1, 1]
        if len(log):
            lvl = step_series(log.sensing_level)
            att = step_series(log.tx_attempt)
            suc = step_series(log.tx_success)
            ax.vlines(t[lvl == 1], 0, 0.8, color="tab:green", alpha=0.5, lw=0.8, label="sense low")
            ax.vlines(t[lvl == 2], 0, 1.0, color="darkgreen", alpha=0.8, lw=0.8, label="sense high")
            mode = step_series(log.tx_mode)
            n_modes = max(1, len(cfg.communication.modes))
            for k in range(n_modes):
                sel = (att == 1) & (suc == 1) & (mode == k)
                if sel.any():
                    ax.scatter(t[sel], np.full(int(sel.sum()), 1.15 + 0.15 * k), marker="^", s=12, color=plt_mode_color(k, n_modes), label=f"tx ok ({cfg.communication.modes[k].name})")
            fail = (att == 1) & (suc == 0)
            ax.scatter(t[fail], np.full(int(fail.sum()), 1.15 + 0.15 * np.clip(mode[fail], 0, n_modes - 1)), marker="x", s=14, color="tab:red", label="tx fail")
            ax.legend(loc="upper center", fontsize=6.5, ncol=3)
        ax.set_ylim(0, 2.4)
        ax.set_yticks([])
        ax.set_title("Sensing and transmission events (tx markers by radio mode)")

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
