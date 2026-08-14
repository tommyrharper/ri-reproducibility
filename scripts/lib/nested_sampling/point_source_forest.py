"""MeqTrees TDL forest: predicts one unpolarized point source's visibilities
into the DATA column of an existing Measurement Set.

Not imported directly. Driven non-interactively by meqtree-pipeliner.py (see
run_meqtrees_predict() in simulate_point_source_ms.py), which loads MS name,
correlation selection, and source flux/position from a generated .tdlconf
file and then runs the "predict" TDL job below.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from Timba.TDL import *
from Timba.Meq import meq
import Meow

mssel = Meow.Context.mssel = Meow.MSUtils.MSSelector(
    has_input=False, has_model=False, has_output=True, tile_sizes=[8, 16, 32], flags=False
)
TDLCompileOptions(*mssel.compile_options())
TDLRuntimeOptions(*mssel.runtime_options())

TDLCompileOption("source_flux_jy", "Point source I flux (Jy)", 1.0, more=float)
TDLCompileOption("source_l_rad", "Point source l offset (rad)", 0.0, more=float)
TDLCompileOption("source_m_rad", "Point source m offset (rad)", 0.0, more=float)


def _define_forest(ns, **kw):
    array, observation = mssel.setup_observation_context(ns)
    # Write straight to DATA: this PoC's MS has no MODEL_DATA/CORRECTED_DATA
    # imaging columns (makems writes only the base columns).
    mssel.output_column = "DATA"

    # At zero offset, reuse the phase-centre Direction object itself (identity
    # check in SkyComponent.visibilities()) rather than building a new
    # LMDirection(0, 0): Meow's K-Jones phase-shift path degenerates at exact
    # l=m=0 and writes a wrongly-shaped result ("shape of child result does
    # not match output column"); the identity-direction path skips K-Jones
    # entirely and is exact for a source at phase centre anyway.
    if source_l_rad == 0.0 and source_m_rad == 0.0:
        direction = observation.phase_centre
    else:
        direction = Meow.LMDirection(ns, "psrc", source_l_rad, source_m_rad)
    # Q/U/V=0.0 (not None) forces Meow to build the full 2x2 brightness matrix.
    # Leaving them None marks the source "unpolarized" and PointSource.brightness()
    # then returns a bare scalar, which Sink can't write into a 2x2 correlation
    # column ("shape of child result does not match output column").
    source = Meow.PointSource(ns, "psrc", direction, I=source_flux_jy, Q=0.0, U=0.0, V=0.0)
    predict = source.visibilities(array, observation)

    Meow.StdTrees.make_sinks(
        ns, predict, array=array, spigots=False, output_col="DATA", corr_index=mssel.get_corr_index()
    )

    TDLRuntimeJob(_predict_job, "Predict point-source visibilities", job_id="predict")


def _predict_job(mqs, parent, wait=False):
    mqs.clearcache("VisDataMux")
    mqs.execute("VisDataMux", mssel.create_io_request(), wait=wait)


if __name__ == "__main__":
    ns = NodeScope()
    _define_forest(ns)
    ns.Resolve()
    print(len(ns.AllNodes()), "nodes defined")
