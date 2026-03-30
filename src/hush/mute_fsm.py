"""Streak-based mute state machine.

Pure logic — no I/O, no GUI, no subprocess. Call update() with each
new predicted label. The FSM returns what to do ('mute', 'unmute', or None);
the caller decides how to act on it.
"""


class MuteFSM:
    def __init__(self, mute_delay=2, unmute_delay=3):
        self.mute_delay = mute_delay
        self.unmute_delay = unmute_delay
        self.muted = False
        self.override = False
        self.ad_streak = 0
        self.ok_streak = 0
        self._paused = False
        # Tracks whether the override was caused by a manual mute (persistent)
        # vs a manual unmute (clears after content run).
        self._override_is_muted = False

    def update(self, pred_label: str, user_unmuted: bool = False,
               user_muted: bool = False) -> str | None:
        """Feed one frame's prediction.

        Args:
            pred_label: 'ad', 'content', or 'neither'.
            user_unmuted: True if we had it muted but the user manually unmuted.
            user_muted: True if we had it unmuted but the user manually muted.

        Returns:
            'mute', 'unmute', or None.
        """
        if self._paused:
            return None

        # User override detection: they manually unmuted while we had it muted
        if self.muted and user_unmuted:
            self.override = True
            self._override_is_muted = False
            self.muted = False
            self.ad_streak = 0
            return None

        # User override detection: they manually muted while we had it unmuted
        if not self.muted and user_muted:
            self.override = True
            self._override_is_muted = True
            self.muted = True
            self.ok_streak = 0
            return None

        # While in override from manual unmute, wait for a content run to re-engage
        if self.override and not self._override_is_muted:
            self._update_streaks(pred_label)
            if self.ok_streak >= self.unmute_delay:
                self.override = False
            return None

        # While in override from manual mute, stay hands-off until user unmutes.
        # Only a user_unmuted event (above) clears this — not a content run.
        if self.override and self._override_is_muted:
            return None

        self._update_streaks(pred_label)

        if not self.muted and self.ad_streak >= self.mute_delay:
            self.muted = True
            return 'mute'

        if self.muted and self.ok_streak >= self.unmute_delay:
            self.muted = False
            return 'unmute'

        return None

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def _update_streaks(self, pred_label: str):
        if pred_label == 'ad':
            self.ad_streak += 1
            self.ok_streak = 0
        elif pred_label == 'content':
            self.ok_streak += 1
            self.ad_streak = 0
        else:  # 'neither' — reset both, hold current state
            self.ad_streak = 0
            self.ok_streak = 0
