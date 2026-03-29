from hush.mute_fsm import MuteFSM


class TestInitialState:
    def test_starts_unmuted(self):
        fsm = MuteFSM()
        assert fsm.muted is False
        assert fsm.override is False
        assert fsm.ad_streak == 0
        assert fsm.ok_streak == 0


class TestMuteTransition:
    def test_mute_after_delay(self):
        fsm = MuteFSM(mute_delay=2)
        assert fsm.update('ad') is None
        assert fsm.update('ad') == 'mute'
        assert fsm.muted is True

    def test_no_mute_below_delay(self):
        fsm = MuteFSM(mute_delay=3)
        assert fsm.update('ad') is None
        assert fsm.update('ad') is None
        assert fsm.muted is False

    def test_mute_with_custom_delay(self):
        fsm = MuteFSM(mute_delay=4)
        for _ in range(3):
            assert fsm.update('ad') is None
        assert fsm.update('ad') == 'mute'


class TestUnmuteTransition:
    def test_unmute_after_delay(self):
        fsm = MuteFSM(mute_delay=1, unmute_delay=3)
        fsm.update('ad')  # mutes
        assert fsm.update('content') is None
        assert fsm.update('content') is None
        assert fsm.update('content') == 'unmute'
        assert fsm.muted is False

    def test_no_unmute_below_delay(self):
        fsm = MuteFSM(mute_delay=1, unmute_delay=3)
        fsm.update('ad')
        fsm.update('content')
        fsm.update('content')
        assert fsm.muted is True


class TestNeither:
    def test_neither_resets_both_streaks(self):
        fsm = MuteFSM(mute_delay=3)
        fsm.update('ad')
        fsm.update('ad')
        # One more ad would mute, but neither resets
        fsm.update('neither')
        assert fsm.ad_streak == 0
        assert fsm.ok_streak == 0
        assert fsm.update('ad') is None  # streak restarted

    def test_neither_holds_muted_state(self):
        fsm = MuteFSM(mute_delay=1, unmute_delay=3)
        fsm.update('ad')  # mutes
        assert fsm.muted is True
        fsm.update('neither')
        assert fsm.muted is True
        assert fsm.ok_streak == 0


class TestStreakReset:
    def test_ad_resets_ok_streak(self):
        fsm = MuteFSM(mute_delay=3, unmute_delay=3)
        fsm.update('content')
        fsm.update('content')
        assert fsm.ok_streak == 2
        fsm.update('ad')
        assert fsm.ok_streak == 0
        assert fsm.ad_streak == 1

    def test_alternating_never_triggers(self):
        fsm = MuteFSM(mute_delay=2, unmute_delay=2)
        for _ in range(20):
            assert fsm.update('ad') is None
            assert fsm.update('content') is None
        assert fsm.muted is False


class TestUserOverride:
    def test_override_stops_muting(self):
        fsm = MuteFSM(mute_delay=1, unmute_delay=2)
        fsm.update('ad')  # mutes
        assert fsm.muted is True
        # User manually unmutes
        fsm.update('ad', user_unmuted=True)
        assert fsm.override is True
        assert fsm.muted is False
        # More ads don't re-mute
        assert fsm.update('ad') is None
        assert fsm.muted is False

    def test_override_clears_after_content_run(self):
        fsm = MuteFSM(mute_delay=1, unmute_delay=2)
        fsm.update('ad')  # mutes
        fsm.update('ad', user_unmuted=True)  # override
        assert fsm.override is True
        fsm.update('content')
        assert fsm.override is True  # not enough yet
        fsm.update('content')
        assert fsm.override is False  # cleared
        # Normal behavior resumes
        assert fsm.update('ad') == 'mute'


class TestPause:
    def test_pause_suppresses_actions(self):
        fsm = MuteFSM(mute_delay=1)
        fsm.pause()
        assert fsm.update('ad') is None
        assert fsm.update('ad') is None
        assert fsm.update('ad') is None
        assert fsm.muted is False

    def test_resume_allows_actions(self):
        fsm = MuteFSM(mute_delay=1)
        fsm.pause()
        fsm.update('ad')
        fsm.update('ad')
        fsm.resume()
        # Streaks didn't accumulate while paused, so need fresh ads
        assert fsm.update('ad') == 'mute'


class TestFullCycle:
    def test_mute_unmute_cycle(self):
        fsm = MuteFSM(mute_delay=2, unmute_delay=3)
        # Content phase
        assert fsm.update('content') is None
        assert fsm.update('content') is None
        # Ad break starts
        assert fsm.update('ad') is None
        assert fsm.update('ad') == 'mute'
        assert fsm.muted is True
        # Ad continues (no action needed, already muted)
        assert fsm.update('ad') is None
        # Content resumes
        assert fsm.update('content') is None
        assert fsm.update('content') is None
        assert fsm.update('content') == 'unmute'
        assert fsm.muted is False
