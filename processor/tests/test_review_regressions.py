"""Independent physical invariants and adversarial checks from the code audit."""
from dataclasses import replace
import math

import numpy as np
import pytest

from geotrace.config import Config, MotionConfig
from geotrace.ekf import ExtendedKalmanFilter
from geotrace.metrics import interpolate_reference, branch_accuracy
from geotrace.models import MotionSample
from geotrace.motion_model import (angular_velocity_from_quaternions, build_imu_stream,
    longitudinal_acceleration, propagate_state, transition_jacobian, process_noise, noise_jacobian)
from geotrace.particle_filter import RoadParticleFilter
from geotrace.pipeline import run_reconstruction, _circular_uncertainty
from geotrace.road_graph import RoadNetwork, build_graph_from_segments


def test_quaternion_sign_changes_do_not_change_angular_velocity():
    t=np.arange(100)*.01
    q=np.column_stack((np.cos(t/2),np.zeros(100),np.zeros(100),np.sin(t/2)))
    changed=q.copy(); changed[20:45]*=-1; changed[70:]*=-1
    assert np.allclose(angular_velocity_from_quaternions(t,q), angular_velocity_from_quaternions(t,changed))
    assert angular_velocity_from_quaternions(t,changed)[:,2] == pytest.approx(np.ones(100),abs=1e-4)


@pytest.mark.parametrize('speed,accel,expected',[(.1,-2.,.0025),(0.,-2.,0.),(44.9,2.,4.4975)])
def test_displacement_integrates_bounded_speed(speed,accel,expected):
    x=np.array([0.,0.,speed,0.,0.,0.]); cfg=MotionConfig()
    out=propagate_state(x,accel,0.,.1,cfg)
    assert out[0] == pytest.approx(expected)


@pytest.mark.parametrize('speed,accel',[(.1,-2.),(44.9,2.),(10.,1.)])
def test_jacobian_includes_world_acceleration_projection(speed,accel):
    cfg=MotionConfig(); x=np.array([0.,0.,speed,.3,.01,.02]); a=np.array([accel,1.,0.]); dt=.1
    along=longitudinal_acceleration(a,x[3])
    analytic=transition_jacobian(x,along,.1,dt,cfg,a_long_heading_derivative=-a[0]*math.sin(x[3])+a[1]*math.cos(x[3]))
    columns=[]
    for i in range(6):
        dx=np.zeros(6);dx[i]=1e-6
        def f(y):return propagate_state(y,longitudinal_acceleration(a,y[3]),.1,dt,cfg)
        columns.append((f(x+dx)-f(x-dx))/(2e-6))
    assert np.allclose(analytic,np.array(columns).T,atol=1e-7)


def test_reanchor_clears_covariance_cross_terms():
    f=ExtendedKalmanFilter(MotionConfig(),[0.,0.,10.,0.,0.,0.])
    f.P=np.eye(6); f.P[3,3]=100.;f.P[3,5]=f.P[5,3]=9.
    f.reanchor((1.,2.),5.,heading_rad=.5,speed=2.)
    assert np.linalg.eigvalsh(f.P).min()>0


def test_integrated_white_noise_is_invariant_to_filter_rate():
    cfg=MotionConfig();x=np.zeros(6)
    def variance(dt):
        g=noise_jacobian(x,dt)
        return (g@process_noise(dt,cfg)@g.T)[2,2]/dt
    assert variance(.1)==pytest.approx(variance(.01))


def motion(t,acc=0.):
    return MotionSample(monotonic_time=t,user_acceleration_g=(acc/9.80665,0.,0.),rotation_rate=(0.,0.,0.))


def test_binning_does_not_fold_samples_outside_requested_window():
    cfg=MotionConfig(robust_window_s=0.,filter_dt_s=.1)
    samples=[motion(-1,100),motion(0),motion(.05),motion(.1),motion(.15),motion(.2),motion(2,100)]
    stream=build_imu_stream(samples,cfg,t_start=0,t_end=.23)
    assert all(c.a_world==(0.,0.,0.) for c in stream.controls)
    assert sum(c.dt for c in stream.controls)==pytest.approx(.23)
    assert stream.controls[-1].t==pytest.approx(.23)


def test_quiet_flag_is_not_backdated_from_future_stop():
    stream=build_imu_stream([motion(i*.01) for i in range(301)],MotionConfig())
    assert not any(c.is_quiet for c in stream.controls if c.t<1.)
    assert any(c.is_quiet for c in stream.controls if c.t>2.)


def short_network():
    graph,frame=build_graph_from_segments([
        (str(i),[(float(i*2),0.),(float(i*2+2),0.)],{'highway':'primary','oneway':True}) for i in range(20)
    ],59.93,30.36)
    return RoadNetwork(graph,frame)


def test_particle_distance_survives_multiple_short_edges():
    net=short_network();cfg=Config();cfg.pf.n_particles=1
    cfg.pf.sigma_s=cfg.pf.sigma_v=cfg.pf.sigma_psi_rad=0.
    cfg.motion.accel_bias_rw=cfg.motion.gyro_bias_rw=0.
    pf=RoadParticleFilter(net,cfg,np.random.default_rng(3));pf.initialize((1.,0.),0.,30.)
    first=min(net.edges,key=lambda e:np.linalg.norm(e.coords[0]))
    pf.edge_idx[:]=first.index;pf.s[:]=1.;pf.v[:]=30.;pf.psi[:]=pf.b_a[:]=pf.b_w[:]=0.
    pf.predict((0.,0.,0.),0.,.5)
    assert pf.positions()[0,0]==pytest.approx(16.,abs=.001)


def test_signed_particle_position_noise_has_no_forward_drift(fork_network):
    cfg=Config();cfg.pf.n_particles=20000;cfg.pf.sigma_s=1.;cfg.pf.sigma_v=0.
    pf=RoadParticleFilter(fork_network,cfg,np.random.default_rng(2));pf.initialize((200.,0.),0.,0.)
    pf.v[:]=pf.b_a[:]=pf.b_w[:]=0.
    before=pf.s.copy();pf.predict((0.,0.,0.),0.,.1)
    assert abs(float(np.mean(pf.s-before)))<.03


def test_log_weights_do_not_turn_bad_gps_into_uniform_belief(fork_network):
    cfg=Config();cfg.pf.n_particles=2
    pf=RoadParticleFilter(fork_network,cfg,np.random.default_rng(2));pf.initialize((200.,0.),0.,0.)
    pf.s[:]=[100.,200.];pf.edge_idx[:]=pf.edge_idx[0];pf.psi[:]=0.;pf.v[:]=0.;pf.w[:]=.5
    pf.update_weights((2000.,0.),gps_sigma=1.,map_evidence_scale=0.)
    assert max(pf.w)>.999
    assert pf.has_diverged()


def test_reference_gaps_are_not_filled_with_invented_truth():
    _,valid=interpolate_reference([0.,1.,100.],np.array([[0.,0.],[1.,0.],[100.,0.]]),[.5,20.,100.])
    assert list(valid)==[True,False,True]


def test_ekf_disk_cannot_count_as_correct_road_branch():
    cfg=Config();f=ExtendedKalmanFilter(cfg.motion)
    disc=_circular_uncertainty(1.,(0.,0.),f,cfg,'LOST',1.,'imu_dead_reckoning')
    metrics=branch_accuracy([disc],[0.,1.],np.zeros((2,2)))
    assert metrics['top1_accuracy'] is None
    assert metrics['evaluated']==0


def test_uncertainty_not_capped_and_responds_to_confidence():
    cfg=Config();f=ExtendedKalmanFilter(cfg.motion);f.P[:2,:2]=np.eye(2)*1e8
    low=_circular_uncertainty(1.,(0.,0.),f,cfg,'LOST',1.,'ekf_dead_reckoning')
    cfg.polygon.confidence=.99
    high=_circular_uncertainty(1.,(0.,0.),f,cfg,'LOST',1.,'ekf_dead_reckoning')
    assert low.total_area_m2>math.pi*2000**2
    assert high.total_area_m2>low.total_area_m2


def test_alignment_never_reads_withheld_gps(tmp_path,monkeypatch):
    import geotrace.live_logs as live
    from test_live_logs import synthetic_drive,write_day
    imu,gps=synthetic_drive();write_day(tmp_path,'2026-07-22',imu,gps)
    captured=[];original=live.estimate_clock_offset
    def capture(geometry,fixes,*args,**kwargs):
        captured.append(fixes.copy());return original(geometry,fixes,*args,**kwargs)
    monkeypatch.setattr(live,'estimate_clock_offset',capture)
    trip,_=live.build_trip(live.ImportSpec(gps_dir=tmp_path/'gps_logs',imu_dir=tmp_path/'imu_logs',day='2026-07-22',gps_warmup_s=80.))
    withheld_raw_time=min(s.wall_time.timestamp()*1000 for s in trip.reference_locations)
    shift=trip.metadata.extra['live_import']['clock_offset']['seconds']*1000
    assert captured[0][-1,0] < withheld_raw_time-shift+.01


def test_realistic_logger_fixture_gyro_agrees_with_attitude():
    from test_live_logs import synthetic_drive,logger_rows
    from geotrace.live_logs import imu_geometry
    imu,_=synthetic_drive();g=imu_geometry(logger_rows(imu))
    rate=np.gradient(np.unwrap(g.yaw),g.t)
    assert np.quantile(abs(rate-g.yaw_rate),.99)<1e-4


def test_combined_flags_and_reference_poisoning_end_to_end(tmp_path):
    from test_live_logs import synthetic_drive,write_day
    from geotrace.live_logs import ImportSpec,build_trip
    imu,gps=synthetic_drive(duration_s=150.);write_day(tmp_path,'2026-07-22',imu,gps)
    trip,_=build_trip(ImportSpec(gps_dir=tmp_path/'gps_logs',imu_dir=tmp_path/'imu_logs',day='2026-07-22',gps_warmup_s=80.))
    # A synthetic map is an explicit fixture, independent of later test poisoning.
    from geotrace.coordinates import LocalFrame
    frame=LocalFrame(gps[0,1],gps[0,2]);xy=frame.to_local_array(gps[::10,1],gps[::10,2])
    graph,frame=build_graph_from_segments([('fixture',xy.tolist(),{'highway':'primary','oneway':True})],frame.lat0,frame.lon0)
    net=RoadNetwork(graph,frame);cfg=Config();cfg.pf.n_particles=80
    cfg.motion.accel_smooth_window_s=5.;cfg.motion.accel_deadband_ms2=2.
    cfg.motion.leveling_recovery_tau_s=20.;cfg.motion.zupt_vibration_g=.02
    cfg.motion.zupt_min_interval_s=5.;cfg.motion.zupt_requires_gps=False
    result=run_reconstruction(trip, net, cfg, algorithm="road_particle_filter")
    poisoned=replace(trip,reference_locations=[replace(s,latitude=s.latitude+1,longitude=s.longitude+1) for s in trip.reference_locations])
    again=run_reconstruction(poisoned, net, cfg, algorithm="road_particle_filter")
    assert np.array_equal(result.primary.array,again.primary.array)
    assert np.isfinite(result.primary.array).all()
    assert np.diff(result.primary.times) == pytest.approx(np.ones(len(result.primary.times)-1), abs=.101)
    assert abs(result.primary.times[-1] - (trip.t0 + trip.duration_s)) <= 1.1
    assert any(s.gps_state=='LOST' for s in result.particle_filter.result.snapshots)
    assert np.array_equal(result.particle_filter.w,again.particle_filter.w)
    # Reporting cadence must not change inference.
    coarse=run_reconstruction(trip,net,cfg,algorithm="road_particle_filter",output_dt=2.)
    assert np.array_equal(result.particle_filter.w,coarse.particle_filter.w)
    assert result.diagnostics['ekf']['final_state'] == coarse.diagnostics['ekf']['final_state']


def test_mount_axis_prevents_lateral_acceleration_from_becoming_forward(fork_network):
    cfg=Config();cfg.pf.n_particles=100
    cfg.pf.sigma_s=cfg.pf.sigma_v=cfg.pf.sigma_psi_rad=0.
    pf=RoadParticleFilter(fork_network,cfg,np.random.default_rng(2));pf.initialize((200.,0.),0.,10.)
    pf.v[:]=10.;pf.b_a[:]=pf.b_w[:]=0.
    pf.predict((3.,0.,0.),0.,.1,a_vehicle=0.)
    assert pf.v == pytest.approx(np.full(100,10.))
    f=ExtendedKalmanFilter(cfg.motion,[0.,0.,10.,0.,0.,0.]);f.predict((3.,0.,0.),0.,.1,a_vehicle=0.)
    assert f.speed==pytest.approx(10.)


def test_junction_proposal_has_importance_correction(fork_network):
    from conftest import edge_named
    cfg=Config();cfg.pf.n_particles=20000
    cfg.pf.sigma_s=cfg.pf.sigma_v=cfg.pf.sigma_psi_rad=0.
    pf=RoadParticleFilter(fork_network,cfg,np.random.default_rng(2));pf.initialize((490.,0.),0.,20.)
    pf.edge_idx[:]=edge_named(fork_network,'Stem',(0.,0.));pf.s[:]=490.
    pf.psi[:]=.5;pf.v[:]=20.;pf.b_a[:]=pf.b_w[:]=0.
    pf.predict((0.,0.,0.),0.,1.)
    a=pf.edge_idx==edge_named(fork_network,'Branch A',(500.,0.))
    assert np.mean(a)>.7  # deliberately gyro-guided proposal
    pf.update_weights(map_evidence_scale=0.)
    assert np.sum(pf.w[a])==pytest.approx(.5,abs=.02)  # equal road priors


def test_gps_recovery_allows_position_noise_at_ten_hz():
    from geotrace.gps_quality import GPSQualityMonitor,GPSState
    from test_gps_quality import fix
    cfg=Config();monitor=GPSQualityMonitor(cfg.gps)
    for i in range(10):
        result=monitor.update(fix(i*.1,accuracy=5.,speed=10.,course=90.),(i+(-1)**i*2.,0.),predicted_speed=10.)
        assert result.accepted,result.reasons
    assert monitor.state is GPSState.TRUSTED


def test_import_calibration_is_invariant_to_poisoned_withheld_rows(tmp_path):
    from test_live_logs import synthetic_drive,write_day
    from geotrace.live_logs import ImportSpec,build_trip
    imu,gps=synthetic_drive();write_day(tmp_path,'2026-07-22',imu,gps)
    spec=ImportSpec(gps_dir=tmp_path/'gps_logs',imu_dir=tmp_path/'imu_logs',day='2026-07-22',gps_warmup_s=80.)
    trip,provenance=build_trip(spec)
    limit=min(s.wall_time.timestamp()*1000 for s in trip.reference_locations)-provenance['clock_offset']['seconds']*1000
    poisoned=gps.copy();mask=gps[:,0]>=limit+.1
    poisoned[mask,1:3]+=1.;poisoned[mask,4]=(poisoned[mask,4]+123.)%360
    write_day(tmp_path,'2026-07-22',imu,poisoned)
    again,info=build_trip(spec)
    assert info['clock_offset']==provenance['clock_offset']
    assert info['mount_estimate']==provenance['mount_estimate']
    assert again.metadata.calibration.forward_axis_device==trip.metadata.calibration.forward_axis_device
    assert [s.to_json() for s in trip.locations]==[s.to_json() for s in again.locations]


def test_physical_gate_stays_speed_bounded_at_the_speed_limit():
    from geotrace.gps_quality import physical_gate
    cfg=Config()
    passed,maximum=physical_gate(10000.,100.,45.,cfg.gps,6.,45.)
    assert not passed
    assert maximum==pytest.approx(4500.+cfg.gps.physical_margin_m)


def test_simulator_attitude_rotates_with_the_vehicle(grid_network):
    from geotrace.simulate import simulate_trip,SimulationSpec
    trip=simulate_trip(grid_network,SimulationSpec(duration_s=140.,gyro_bias_rads=0.,gyro_noise_rads=0.))
    t=np.array([s.monotonic_time for s in trip.motions])
    q=np.array([s.quaternion for s in trip.motions])
    gyro=np.array([s.rotation_rate for s in trip.motions])
    implied=angular_velocity_from_quaternions(t,q)
    assert np.max(abs(implied[1:-1]-gyro[1:-1]))<.01


def test_shock_coasting_does_not_integrate_sensor_bias():
    cfg=MotionConfig();f=ExtendedKalmanFilter(cfg,[0.,0.,10.,.3,1.,.2])
    f.predict((0.,0.,0.),0.,.1,coast=True)
    assert f.speed==pytest.approx(10.)
    assert f.heading==pytest.approx(.3)
