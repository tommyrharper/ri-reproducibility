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
    # makems writes only base columns, so write predictions straight to DATA.
    mssel.output_column = "DATA"

    # K-Jones degenerates at l=m=0; phase_centre skips it exactly there.
    if source_l_rad == 0.0 and source_m_rad == 0.0:
        direction = observation.phase_centre
    else:
        direction = Meow.LMDirection(ns, "psrc", source_l_rad, source_m_rad)
    # Q/U/V=0.0 forces the 2x2 brightness matrix Sink requires.
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
