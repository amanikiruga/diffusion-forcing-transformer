from algorithms.dfot.history_guidance import HistoryGuidance as _HistoryGuidance

class HistoryGuidance(_HistoryGuidance):
    @classmethod
    def smart(
        cls,
        x_angle: float,
        y_angle: float,
        distance: float,
        visualize: bool = False,
    ):
        if abs(x_angle) < 30 and abs(y_angle) < 30 and distance < 150:
            return cls.stabilized_fractional(
                guidance_scale=4.0,
                stabilization_level=0.02,
                freq_scale=0.4,
                visualize=visualize,
            )
        else:
            return cls.stabilized_vanilla(
                guidance_scale=4.0,
                stabilization_level=0.02,
                visualize=visualize,
            )