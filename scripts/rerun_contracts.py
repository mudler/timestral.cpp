"""Record selection shared by the rerun harness and its artifact consumers."""
import re

# Ordinary names describe isolated operations. block_refined is the sole alias:
# predict_block + refine on block arguments, compared against gold refined.
_BLOCK = re.compile(r'^(pos|neg)\.b\d+\.(scaled|loc|scale|known\.(loc|scale|is_binary|scaled)|'
                    r'patch\.(hidden_out|out)|layer\.\d+\.(out|mlp_norm|mlp|attn\.\d+\..+)|'
                    r'out_norm|head\.(raw|monotone)|predict|refined|frames\.(loc|scale|in\.sinh_median))$')


def expected_reruns(source_names):
    """Map required rerun output names to gold comparison names, independently of results."""
    result = {n: n for n in source_names
              if _BLOCK.match(n) or re.match(r'^(pos|neg)\.cap\.(out|scale)$', n)}
    for n in list(result):
        if n.endswith('.refined'):
            result[n[:-len('refined')] + 'block_refined'] = n
    return result


def source_name(name):
    return name[:-len('block_refined')] + 'refined' if name.endswith('.block_refined') else name


def floor_inputs(rerun):
    """Select combined block refinement for the gold refined floor; fail on old artifacts."""
    selected = dict(rerun)
    for name in rerun:
        if name.endswith('.refined'):
            combined = name[:-len('refined')] + 'block_refined'
            if combined not in rerun:
                raise ValueError(f'{combined}: combined block contract missing; regenerate this rerun')
            selected[name] = rerun[combined]
    return selected
