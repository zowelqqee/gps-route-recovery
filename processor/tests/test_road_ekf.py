"""Physical and probabilistic contracts for the road-coordinate model."""
from dataclasses import replace
import math

import numpy as np
import pytest
from shapely.geometry import LineString
from scipy.special import ndtr

from geotrace.config import Config
from geotrace.motion_model import ImuControl
from geotrace.road_ekf import RoadEKF, Mode, truncate, scalar_update, S, PSI, BW, ROAD_ERROR
from geotrace.road_graph import RoadNetwork, build_graph_from_segments
from geotrace.pipeline import run_reconstruction, build_metrics
from geotrace.simulate import simulate_trip, SimulationSpec
from geotrace.fault_injection import inject_faults, FaultSpec
from conftest import edge_named


def straight():
    return RoadNetwork(*build_graph_from_segments([
        ('road', [(0.,0.), (5000.,0.)], {'highway':'primary','oneway':True})],59.9,30.3))


def test_truncated_mixture_conserves_all_state_moments():
    rng=np.random.default_rng(43)
    A=rng.normal(size=(6,6));P=A@A.T
    m=Mode(0,np.array([10.,5.,.2,.1,.01,0.]),P,0.,(0,))
    # A partition is not a measurement. Combining its modes must recover P.
    pieces=[truncate(m,a,b) for a,b in [(-math.inf,9.),(9.,12.),(12.,math.inf)]]
    w=np.array([math.exp(p.log_weight) for p in pieces])
    mean=sum(wi*p.x for wi,p in zip(w,pieces))
    cov=sum(wi*(p.P+np.outer(p.x-mean,p.x-mean)) for wi,p in zip(w,pieces))
    # Heading wraps; this test stays far from the angular branch cut.
    assert w.sum()==pytest.approx(1.)
    assert mean==pytest.approx(m.x)
    assert cov==pytest.approx(P)


def test_uncertainty_is_a_small_road_subset_and_reports_omitted_mass():
    net=straight();cfg=Config();bank=RoadEKF(net,cfg)
    bank.seed((1000.,0.),10.,0.,3.)
    bank.best.P[S,S]=1000.**2
    u=bank.uncertainty(500,'LOST',500)
    assert u.status=='AMBIGUOUS'
    assert u.represented_mass==pytest.approx(ndtr(.03)-ndtr(-.03))
    for c in u.components:
        assert c.geometry.difference(net.edges[c.edge_indices[0]].line.buffer(6.000001)).area < 1e-5
        assert c.geometry.area <= 60*12+math.pi*6**2
    assert not u.contains((1000.,100.))
    assert u.to_json(net.frame)['represented_mass']==pytest.approx(u.represented_mass)


def test_straight_road_constraint_does_not_invent_distance_information():
    net=straight();cfg=Config();bank=RoadEKF(net,cfg)
    bank.seed((1000.,0.),10.,0.,3.)
    initial=bank.best.P[S,S]
    for i in range(10):bank.predict(ImuControl(i*.5,.5,0.,0.))
    weights=np.exp([m.log_weight for m in bank.modes])
    mean=sum(w*m.x for w,m in zip(weights,bank.modes))
    variance=sum(w*(m.P[S,S]+(m.x[S]-mean[S])**2) for w,m in zip(weights,bank.modes))
    assert mean[S]==pytest.approx(1050.,abs=.05)
    assert variance>initial+50


def test_turn_constraint_updates_distance_via_curvature():
    P=np.diag([100.,4.,.1,.01,.0001,.01])
    m=Mode(0,np.array([10.,5.,0.,.4,0.,0.]),P,0.,(0,))
    # psi(s)=0.02s, observed heading=.4 implies a later position on the bend.
    H=np.array([-.02,0.,0.,1.,0.,-1.])
    scalar_update(m, .2-.4,H,.01)
    assert m.x[S]>10.
    assert m.P[S,S]<100.
    assert np.linalg.eigvalsh(m.P).min()>0


def test_gap_stops_road_estimate_instead_of_creating_a_disc():
    bank=RoadEKF(straight(),Config());bank.seed((100.,0.),10.,0.,3.)
    bank.predict(ImuControl(10.,10.,0.,0.,gap_exceeded=True))
    u=bank.uncertainty(10,'LOST',10)
    assert bank.best is None
    assert u.status=='LOST' and not u.components and u.total_area_m2==0


def test_short_edges_preserve_distance_and_one_way_direction():
    net=RoadNetwork(*build_graph_from_segments([
        (str(i),[(i*2.,0.),((i+1)*2.,0.)],{'oneway':True,'highway':'primary'})
        for i in range(40)],59.9,30.3))
    bank=RoadEKF(net,Config());bank.seed((1.,0.),20.,0.,.01)
    bank.modes=[bank.best];bank.best.log_weight=0.;bank.best.P=np.eye(6)*1e-8
    bank.predict(ImuControl(.5,.5,0.,0.))
    assert bank.position()[0]==pytest.approx(11.,abs=.1)
    assert all(net.edges[m.edge].coords[-1,0]>net.edges[m.edge].coords[0,0] for m in bank.modes)


def test_default_pipeline_selects_gyro_branch_and_never_reads_reference(fork_network):
    net=fork_network
    trip=simulate_trip(net,SimulationSpec(duration_s=100.,cruise_speed_ms=9.,warmup_still_s=5.),
        seed=42,route=[edge_named(net,'Stem',(0.,0.)),edge_named(net,'Branch A',(500.,0.))])
    broken,_=inject_faults(trip,[FaultSpec(kind='dropout',start_s=40.,duration_s=60.)],seed=42,frame=net.frame)
    cfg=Config()
    result=run_reconstruction(broken,net,cfg)
    assert result.algorithm=='road_ekf' and result.particle_filter is None
    final=result.uncertainty[-1]
    assert final.best and 'Branch A' in final.best.street_names
    assert result.primary.xy[-1][1]>100.
    assert all(u.total_area_m2<4200 for u in result.uncertainty)
    assert all(net.distance_to_road(p)<1e-5 for p in result.primary.xy)
    # Poison every withheld fix. Inference must be bit-for-bit unchanged.
    poisoned=replace(broken,reference_locations=[replace(s,latitude=s.latitude+1.) for s in broken.reference_locations])
    again=run_reconstruction(poisoned,net,cfg)
    assert again.primary.xy==result.primary.xy
    assert [u.to_json(net.frame) for u in again.uncertainty]==[u.to_json(net.frame) for u in result.uncertainty]
    assert build_metrics(broken,result,cfg).position_error['mean_m']<60
    for segment in result.primary.to_geojson(net.frame)['geometry']['coordinates']:
        xy=net.frame.to_local_array([c[1] for c in segment],[c[0] for c in segment])
        if len(xy)>1:
            road=LineString(xy)
            assert all(net.distance_to_road(road.interpolate(x).coords[0])<1e-4 for x in np.linspace(0,road.length,30))


def test_report_handles_rejected_gps_and_uses_road_geometry(fork_network,tmp_path):
    from geotrace.visualization import ReportInputs, build_report
    trip=simulate_trip(fork_network,SimulationSpec(duration_s=8.,warmup_still_s=1.),seed=12,
        route=[edge_named(fork_network,'Stem',(0.,0.))])
    trip.locations[3]=replace(trip.locations[3],latitude=trip.locations[3].latitude+.1)
    cfg=Config();result=run_reconstruction(trip,fork_network,cfg)
    assert result.diagnostics['rejected_fixes']
    report=build_report(ReportInputs(trip,result,build_metrics(trip,result,cfg).to_json()),tmp_path/'report.html')
    content=report.read_text()
    assert 'Local road hypotheses' in content
    assert '95% nominal position regions' not in content
    assert 'no position is available' in content


def test_missing_map_support_is_not_reported_as_a_successful_branch():
    from geotrace.metrics import branch_accuracy
    bank=RoadEKF(straight(),Config())
    u=bank.uncertainty(10.,'LOST',10.)
    metrics=branch_accuracy([u],[10.],np.array([[100.,0.]]))
    assert metrics['evaluated']==1 and metrics['top1_accuracy']==0.


def test_speed_floor_does_not_erase_moving_probability():
    bank=RoadEKF(straight(),Config());bank.seed((1000.,0.),0.,0.,1.)
    bank.modes=[bank.best];bank.best.log_weight=0.
    bank.best.P=np.diag([1.,4.,1e-8,1e-8,1e-12,1e-8])
    bank.predict(ImuControl(.5,.5,0.,0.))
    # Half of an uncertain zero-mean speed lies above zero. A clipped-mean
    # Jacobian incorrectly makes all of it a stationary certainty.
    weights=np.exp([m.log_weight for m in bank.modes])
    speed=sum(w*m.x[1] for w,m in zip(weights,bank.modes))
    variance=sum(w*(m.P[1,1]+(m.x[1]-speed)**2) for w,m in zip(weights,bank.modes))
    assert speed>.3 and variance>1.
    assert sum(w*m.x[S] for w,m in zip(weights,bank.modes))>1000.1
    assert all(np.linalg.eigvalsh(m.P).min()>0 for m in bank.modes)


def test_accelerometer_bump_does_not_delete_a_real_gyro_turn():
    cfg=Config();net=straight()
    normal=RoadEKF(net,cfg);bump=RoadEKF(net,cfg)
    for bank in (normal,bump):
        bank.seed((1000.,0.),10.,0.,1.)
    normal.predict(ImuControl(.5,.5,0.,.3,peak_gyro_rads=.32))
    bump.predict(ImuControl(.5,.5,0.,.3,is_shock=True,yaw_trust=.5,
                            peak_accel_ms2=10.6,peak_gyro_rads=.32))
    assert bump.best.x[PSI]==pytest.approx(normal.best.x[PSI],abs=1e-7)
    assert bump.best.x[PSI]>0.
