"""Small scheduling policy for keeping per-frame hand work ahead of discovery."""
import time


class WorkPriority:
    """Defer best-effort discovery briefly when current hand work is due."""
    def __init__(self, clock=time.monotonic, hand_grace_seconds=.15,
                 maximum_deferral_seconds=1.):
        self.clock = clock
        self.hand_grace_seconds = hand_grace_seconds
        self.maximum_deferral_seconds = maximum_deferral_seconds
        self.hand_priority_until = float('-inf')
        self.discovery_deferred_since = None

    def observe_hand_update(self, hand_result, now=None):
        """Reserve a short window after a hand refresh/reacquire is requested."""
        now = self.clock() if now is None else now
        scheduler = hand_result.get('scheduler', {}) if hand_result else {}
        if (scheduler.get('inference_busy') or
                scheduler.get('request_reason') not in (None, 'stable_tracking')):
            self.hand_priority_until = max(self.hand_priority_until,
                                           now+self.hand_grace_seconds)

    def allow_background_discovery(self, hand_pipeline=None, now=None):
        """Discovery is best effort, but repeated hand work cannot starve it."""
        now = self.clock() if now is None else now
        next_detection_at = getattr(hand_pipeline, 'next_detection_at', float('inf'))
        hand_due = (hand_pipeline is not None and
                    (getattr(hand_pipeline, 'track', None) is None or
                     isinstance(next_detection_at, (int, float)) and now >= next_detection_at))
        hand_priority = (getattr(hand_pipeline, 'busy', False) or hand_due or
                         now < self.hand_priority_until)
        if not hand_priority:
            self.discovery_deferred_since = None
            return True
        if self.discovery_deferred_since is None:
            self.discovery_deferred_since = now
        return now-self.discovery_deferred_since >= self.maximum_deferral_seconds
