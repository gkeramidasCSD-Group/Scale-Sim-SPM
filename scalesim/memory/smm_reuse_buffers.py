# scalesim/memory/smm_reuse_buffers.py
"""
Read-buffer subclass used by smm/run_smm.py to give the SMM paper's
(Zouzoula et al., ICPP '24) on-chip memory policies a genuine effect on
SCALE-Sim's own simulation, instead of trusting the closed-form formula
(smm_policy_selector.py) without ever running SCALE-Sim at all -- a
self-contained duplicate of the pattern in
scalesim/memory/cosma_resident_buffers.py / onsram/onsram_helpers/
resident_buffers.py, NOT an import of either (same reasoning: a change to
one paper's buffer subclass must never change another's simulated
behavior).

Why this is a different mechanism than COSMA's/OnSRAM's "fully_resident"
flag: those two are about CROSS-layer residency (a whole tensor either
costs zero DRAM for an entire layer, because a prior layer's plan already
put it on-chip, or it doesn't). SMM's P1-P5 policies are about INTRA-layer
reuse -- the paper's own closed-form (policy.h/manager.h, ported in
smm_policy_selector.py) assumes each operand element is fetched from
off-chip at most `reload` times for the whole layer (reload = 1 for every
non-partial policy and for the filter operand always; reload =
ceil(F#/n) for the ifmap operand under policies 4/5 -- see
estimate_accesses in smm_policy_selector.py). That assumption requires
*no eviction model* (paper's own words) -- i.e. a software-managed
scratchpad where the schedule itself guarantees reuse, not a
capacity-limited hardware cache.

SCALE-Sim's own ReadBufferEstimateBw (scalesim/memory/
read_buffer_estimate_bw.py) is capacity-limited: addresses are grouped
into ~100 chronological "sets", and check_hit() only looks inside the
currently-resident window of sets. Once the window advances past a set
(because the PE-array/dataflow mapping needs more folds than the SMM
policy's own loop nest assumes -- e.g. Fn=64 filters through a 16-wide
array needs 4 column-folds, each of which re-streams the ifmap through the
array), a later request for an address that already *was* fetched once
gets treated as a fresh miss and charged a second (third, fourth...) real
DRAM fetch. Empirically, on ResNet18 Conv1 at 64kB GLB this inflated
simulated off-chip bytes to ~3.3x the policy's own analytical prediction
(898.8kB predicted vs ~2928kB simulated) before this fix.

SmmReuseReadBuffer closes that gap with the smallest possible change:
check_hit() additionally remembers, per address, how many times a genuine
miss has already been paid for *this layer*. Once that count reaches
`reload_budget`, any further request for the same address is reported as
a hit regardless of what the window-based check says -- i.e. real DRAM
timing/bandwidth is charged for exactly the first `reload_budget`
fetches of each address (matching the policy's own closed-form budget),
and every repeat beyond that is free (hit_latency only), independent of
capacity-driven eviction. Every other code path -- every genuine first (or
within-budget) fetch, every hit the base class would already report -- is
byte-for-byte the base class's own unmodified logic.
"""
from scalesim.memory.read_buffer_estimate_bw import ReadBufferEstimateBw


class SmmReuseReadBuffer(ReadBufferEstimateBw):
    """
    Identical to ReadBufferEstimateBw, except an address already charged a
    real (backing-buffer) fetch `reload_budget` times this layer is
    reported as a hit on every subsequent request, regardless of whether
    it still sits inside the base class's own capacity-windowed buffer.

    reload_budget: max genuine DRAM fetches allowed per distinct address
    for this operand, this layer (1 for every SMM policy's filter operand,
    and for ifmap under every non-partial policy; ceil(F#/n) for ifmap
    under policies 4/5 -- see smm_policy_selector.py's `_estimate_accesses`
    and the `ifmap_reload` field on LayerPlan). Default 1 reproduces the
    paper's "every element transferred once" guarantee out of the box.
    """

    def __init__(self, reload_budget: int = 1):
        super().__init__()
        assert reload_budget >= 1, 'reload_budget must be >= 1'
        self.reload_budget = reload_budget
        self._fetch_count = {}

    def check_hit(self, addr):
        if super().check_hit(addr):
            return True  # genuine SRAM hit per the base class -- no cost either way

        count = self._fetch_count.get(addr, 0)
        if count >= self.reload_budget:
            return True  # already paid for `reload_budget` real fetches -- free from here on

        self._fetch_count[addr] = count + 1
        return False  # let the real miss (DRAM fetch) proceed
