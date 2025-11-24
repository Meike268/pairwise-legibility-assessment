from scipy.stats import kendalltau


class Evaluator:
    """
    Compare two rankings and report evaluation metrics.

    The evaluator expects two ordered lists of item IDs (best-to-worst) and a
    complete set/list of all IDs to align the rank vectors before computing
    metrics.

    Args:
        ranking1 (list): Ordered list of IDs (best -> worst) for the first ranking.
        ranking2 (list): Ordered list of IDs (best -> worst) for the second ranking.
        all_ids (iterable): Iterable containing every ID present in the rankings
            in a fixed order used to align rank vectors.
    """
    def __init__(self, ranking1, ranking2, all_ids):
        self.ranking1 = ranking1
        self.ranking2 = ranking2
        self.all_ids = all_ids

    def kendall_tau_from_rankings(self):
        """
        Compute Kendall's tau correlation between the two stored rankings.

        The method converts each ranking into a numeric rank vector aligned to
        ``self.all_ids`` and then computes the Kendall tau and a two-sided
        p-value using SciPy's implementation.

        Returns:
            tuple: (tau, p_value) where ``tau`` is Kendall's tau correlation
                coefficient (float) and ``p_value`` is the two-sided p-value.
        """

        def ranking_to_vector(ranking, all_ids):
            """Convert an ordered ID list into a rank vector aligned with all_ids.

            Args:
                ranking (list): Ordered list of IDs (best -> worst).
                all_ids (iterable): Iterable of all IDs defining the target order.

            Returns:
                list[int]: Numeric ranks corresponding to each id in ``all_ids``.
            """

            rank_map = {id_: rank for rank, id_ in enumerate(ranking, start=1)}
            return [rank_map[id_] for id_ in all_ids]

        ranks1 = ranking_to_vector(self.ranking1, self.all_ids)
        ranks2 = ranking_to_vector(self.ranking2, self.all_ids)

        tau, p_value = kendalltau(ranks1, ranks2)
        return tau, p_value





