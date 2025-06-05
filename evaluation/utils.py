""" Utilities for formatting data for comet and metricX """

"""Format data for comet models"""
def format_kiwi(sources: list[str], hypotheses: list[str]) -> list[dict[str, str]]:
    data = []

    for src, hyp in zip(sources, hypotheses):
        entry = {
            "src": src,
            "mt": hyp,
        }
        data.append(entry)

    return data


def format_comet(sources: list[str], hypotheses: list[str], references: list[str]) -> list[dict[str, str]]:
    """Format all sents in a corpus"""
    data = []

    for src, hyp, ref in zip(sources, hypotheses, references):
        entry = {
            "src": src,
            "mt": hyp,
            "ref": ref,
        }
        data.append(entry)

    return data


""" MetricX formatting """
def format_metricx(sources: list[str], hypotheses: list[str], references: list[str]) -> list[dict[str, str]]:
    data = []

    for src, hyp, ref in zip(sources, hypotheses, references):
        entry = {
            "source": src,
            "hypothesis": hyp,
            "reference": ref,
        }
        data.append(entry)

    return data

def format_metricx_qe(sources: list[str], hypotheses: list[str]) -> list[dict[str, str]]:
    data = []

    for src, hyp in zip(sources, hypotheses):
        entry = {
            "source": src,
            "hypothesis": hyp,
        }
        data.append(entry)

    return data
