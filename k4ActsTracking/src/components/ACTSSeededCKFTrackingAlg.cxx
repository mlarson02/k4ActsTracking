/*
 * Copyright (c) 2014-2024 Key4hep-Project.
 *
 * This file is part of Key4hep.
 * See https://key4hep.github.io/key4hep-doc/ for further info.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "k4ActsTracking/ACTSSeededCKFTrackingAlg.hxx"

// ACTSTracking
#include "k4ActsTracking/MeasurementCalibrator.hxx"

// edm4hep
#include <edm4hep/MCParticle.h>
#include <edm4hep/MutableTrack.h>
#include <edm4hep/SimTrackerHit.h>
#include <edm4hep/TrackState.h>
#include <edm4hep/TrackerHitPlane.h>

// ACTS
#include <Acts/Seeding/SpacePointGrid.hpp>
#include <Acts/Surfaces/PerigeeSurface.hpp>
#include <Acts/TrackFinding/CombinatorialKalmanFilter.hpp>
#include <Acts/TrackFinding/MeasurementSelector.hpp>
#include <Acts/TrackFinding/TrackStateCreator.hpp>
#include <Acts/TrackFitting/GainMatrixUpdater.hpp>
#include <Acts/Utilities/RangeXD.hpp>

// TBB
#include <tbb/concurrent_vector.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>
#include <tbb/task_arena.h>

#include <chrono>

using namespace Acts::UnitLiterals;

DECLARE_COMPONENT(ACTSSeededCKFTrackingAlg)

// Constructor
ACTSSeededCKFTrackingAlg::ACTSSeededCKFTrackingAlg(const std::string& name, ISvcLocator* svcLoc)
    : ACTSAlgBase(name, svcLoc) {}

StatusCode ACTSSeededCKFTrackingAlg::initialize() {
  // Initialize the base
  StatusCode init = ACTSAlgBase::initialize();

  // Initialize seeding layers
  std::vector<std::string> seedingLayers;
  std::copy_if(m_seedingLayers.begin(), m_seedingLayers.end(), std::back_inserter(seedingLayers),
               [](const std::string& s) { return !s.empty(); });

  if (seedingLayers.size() % 2 != 0) {
    throw std::runtime_error("SeedingLayers needs an even number of entries");
  }

  std::vector<Acts::GeometryIdentifier> geoSelection;
  for (uint32_t i = 0; i < seedingLayers.size(); i += 2) {
    Acts::GeometryIdentifier geoid;
    if (m_seedingLayers[i + 0] != "*")  // volume
      geoid = geoid.withVolume(std::stoi(m_seedingLayers[i + 0]));
    if (m_seedingLayers[i + 1] != "*")  // layer
      geoid = geoid.withLayer(std::stoi(m_seedingLayers[i + 1]));

    geoSelection.push_back(geoid);
  }

  m_seedGeometrySelection = ACTSTracking::GeometryIdSelector(geoSelection);

  if (m_seedFinding_deltaRMinTop == 0.f)
    m_seedFinding_deltaRMinTop = m_seedFinding_deltaRMin;
  if (m_seedFinding_deltaRMaxTop == 0.f)
    m_seedFinding_deltaRMaxTop = m_seedFinding_deltaRMax;
  if (m_seedFinding_deltaRMinBottom == 0.f)
    m_seedFinding_deltaRMinBottom = m_seedFinding_deltaRMin;
  if (m_seedFinding_deltaRMaxBottom == 0.f)
    m_seedFinding_deltaRMaxBottom = m_seedFinding_deltaRMax;

  debug() << "Initialization of ACTSSeededCKFTrackingAlg successful" << endmsg;
  return init;
}

std::tuple<edm4hep::TrackCollection, edm4hep::TrackCollection> ACTSSeededCKFTrackingAlg::operator()(
    const edm4hep::TrackerHitPlaneCollection& trackerHitCollection) const {
  // Prepare output collections
  edm4hep::TrackCollection seedCollection;
  edm4hep::TrackCollection trackCollection;

  // Containers
  std::vector<std::pair<Acts::GeometryIdentifier, edm4hep::TrackerHitPlane>> sortedHits;
  ACTSTracking::SourceLinkContainer                                          sourceLinks;
  ACTSTracking::MeasurementContainer                                         measurements;
  ACTSTracking::SeedSpacePointContainer                                      spacePoints;

  // Loop over each hit collections and get a single vector with hits
  // from all of the subdetectors. Also include the Acts GeoId in
  // the vector. It will be important for the sort to speed up the
  // population of the final SourceLink multiset.
  sortedHits.reserve(trackerHitCollection.size());

  for (const auto& hit : trackerHitCollection) {
    sortedHits.push_back(std::make_pair(geoIDMappingTool()->getGeometryID(hit), hit));
  }
  debug() << "Working with " << sortedHits.size() << " hits." << endmsg;

  // Sort by GeoID
  auto compare = [](const auto& a, const auto& b) { return a.first < b.first; };

  tbb::task_arena arena(m_numThreads.value());
  std::cout << "m_numThreads: " << m_numThreads << "\n";

  if (m_numThreads > 1) {
    arena.execute([&] { tbb::parallel_sort(sortedHits.begin(), sortedHits.end(), compare); });
  } else {
    std::sort(sortedHits.begin(), sortedHits.end(), compare);
  }

  // Turn the edm4hep TrackerHit's into Acts objects
  // Assumes that the hits are sorted by the GeoID
  sourceLinks.reserve(sortedHits.size());
  for (auto& hitPair : sortedHits) {
    // Convert to Acts hit
    const Acts::Surface* surface = trackingGeometry()->findSurface(hitPair.first);
    if (surface == nullptr)
      throw std::runtime_error("Surface not found");

    const edm4hep::Vector3d& edmglobalpos = hitPair.second.getPosition();
    Acts::Vector3            globalPos    = {edmglobalpos.x, edmglobalpos.y, edmglobalpos.z};

    Acts::Result<Acts::Vector2> lpResult = surface->globalToLocal(geometryContext(), globalPos, {0, 0, 0}, 0.5_um);
    if (!lpResult.ok())
      throw std::runtime_error("Global to local transformation did not succeed.");

    Acts::Vector2 loc = lpResult.value();

    Acts::SquareMatrix2            localCov = Acts::SquareMatrix2::Zero();
    const edm4hep::TrackerHitPlane hitplane = hitPair.second;
    localCov(0, 0)                          = std::pow(hitplane.getDu() * Acts::UnitConstants::mm, 2);
    localCov(1, 1)                          = std::pow(hitplane.getDv() * Acts::UnitConstants::mm, 2);

    ACTSTracking::SourceLink  sourceLink(surface->geometryId(), measurements.size(), hitPair.second);
    Acts::SourceLink          src_wrap{sourceLink};
    ACTSTracking::Measurement meas =
        ACTSTracking::makeMeasurement(src_wrap, loc, localCov, Acts::eBoundLoc0, Acts::eBoundLoc1);

    measurements.push_back(meas);
    sourceLinks.emplace_hint(sourceLinks.end(), sourceLink);

    // Seed selection and conversion to useful coordinates
    if (m_seedGeometrySelection.check(surface->geometryId())) {
      Acts::RotationMatrix3 rotLocalToGlobal = surface->referenceFrame(geometryContext(), globalPos, {0, 0, 0});

      // Convert to a seed space point
      // the space point requires only the variance of the transverse and
      // longitudinal position. reduce computations by transforming the
      // covariance directly from local to rho/z.
      //
      // compute Jacobian from global coordinates to rho/z
      //
      //         rho = sqrt(x² + y²)
      // drho/d{x,y} = (1 / sqrt(x² + y²)) * 2 * {x,y}
      //             = 2 * {x,y} / r
      //       dz/dz = 1 (duuh!)

      double             x            = globalPos[Acts::ePos0];
      double             y            = globalPos[Acts::ePos1];
      double             scale        = 2 / std::hypot(x, y);
      Acts::Matrix<2, 3> jacXyzToRhoZ = Acts::Matrix<2, 3>::Zero();
      jacXyzToRhoZ(0, Acts::ePos0)    = scale * x;
      jacXyzToRhoZ(0, Acts::ePos1)    = scale * y;
      jacXyzToRhoZ(1, Acts::ePos2)    = 1;
      // compute Jacobian from local coordinates to rho/z
      const auto jac = jacXyzToRhoZ * rotLocalToGlobal.block<3, 2>(Acts::ePos0, Acts::ePos0);
      // compute rho/z variance
      const auto var = (jac * localCov * jac.transpose()).diagonal();

      // Save spacepoint
      spacePoints.push_back(ACTSTracking::SeedSpacePoint(globalPos, var[0], var[1], sourceLink));
    }
  }

  debug() << "Created " << spacePoints.size() << " space points" << endmsg;

  // Run seeding + tracking algorithms
  // Caches
  Acts::MagneticFieldContext         magFieldContext = Acts::MagneticFieldContext();
  Acts::MagneticFieldProvider::Cache magCache        = magneticField()->makeCache(magFieldContext);

  // std::unique_ptr<const Acts::Logger>
  // logger=Acts::getDefaultLogger("TrackFitting",
  // Acts::Logging::Level::VERBOSE);

  // Finder configuration
  static const Acts::Vector3 zeropos(0, 0, 0);

  Acts::SeedFinderConfig<SSPoint> finderCfg;
  finderCfg.rMax                     = m_seedFinding_rMax;
  finderCfg.deltaRMin                = m_seedFinding_deltaRMin;
  finderCfg.deltaRMax                = m_seedFinding_deltaRMax;
  finderCfg.deltaRMinTopSP           = m_seedFinding_deltaRMinTop;
  finderCfg.deltaRMaxTopSP           = m_seedFinding_deltaRMaxTop;
  finderCfg.deltaRMinBottomSP        = m_seedFinding_deltaRMinBottom;
  finderCfg.deltaRMaxBottomSP        = m_seedFinding_deltaRMaxBottom;
  finderCfg.collisionRegionMin       = -m_seedFinding_collisionRegion;
  finderCfg.collisionRegionMax       = m_seedFinding_collisionRegion;
  finderCfg.zMin                     = -m_seedFinding_zMax;
  finderCfg.zMax                     = m_seedFinding_zMax;
  finderCfg.maxSeedsPerSpM           = 1;
  finderCfg.cotThetaMax              = 7.40627;  // 2.7 eta;
  finderCfg.sigmaScattering          = m_seedFinding_sigmaScattering;
  finderCfg.radLengthPerSeed         = m_seedFinding_radLengthPerSeed;
  finderCfg.minPt                    = m_seedFinding_minPt * Acts::UnitConstants::MeV;
  finderCfg.impactMax                = m_seedFinding_impactMax * Acts::UnitConstants::mm;
  finderCfg.useVariableMiddleSPRange = true;

  Acts::SeedFilterConfig filterCfg;
  filterCfg.maxSeedsPerSpM = finderCfg.maxSeedsPerSpM;

  finderCfg.seedFilter = std::make_unique<Acts::SeedFilter<SSPoint>>(filterCfg);
  finderCfg            = finderCfg.calculateDerivedQuantities();

  Acts::SeedFinderOptions finderOpts;
  finderOpts.bFieldInZ = (*magneticField()->getField(zeropos, magCache))[2];
  finderOpts.beamPos   = {0, 0};
  finderOpts           = finderOpts.calculateDerivedQuantities(finderCfg);

  Acts::CylindricalSpacePointGridConfig gridCfg;
  gridCfg.cotThetaMax = finderCfg.cotThetaMax;
  gridCfg.deltaRMax   = finderCfg.deltaRMax;
  gridCfg.minPt       = finderCfg.minPt;
  gridCfg.rMax        = finderCfg.rMax;
  gridCfg.zMax        = finderCfg.zMax;
  gridCfg.zMin        = finderCfg.zMin;
  gridCfg.impactMax   = finderCfg.impactMax;
  if (m_seedFinding_zBinEdges.size() > 0) {
    gridCfg.zBinEdges.resize(m_seedFinding_zBinEdges.size());
    for (size_t k = 0; k < m_seedFinding_zBinEdges.size(); k++) {
      float pos = std::atof(m_seedFinding_zBinEdges[k].c_str());
      if (pos >= finderCfg.zMin && pos < finderCfg.zMax) {
        gridCfg.zBinEdges[k] = pos;
      } else {
        warning() << "Wrong parameter SeedFinding_zBinEdges; "
                  << "default used" << endmsg;
        gridCfg.zBinEdges.clear();
        break;
      }
    }
  }

  Acts::CylindricalSpacePointGridOptions gridOpts;
  gridOpts.bFieldInZ = (*magneticField()->getField(zeropos, magCache))[2];

  std::vector<const ACTSTracking::SeedSpacePoint*> spacePointPtrs(spacePoints.size(), nullptr);
  std::transform(spacePoints.begin(), spacePoints.end(), spacePointPtrs.begin(),
                 [](const ACTSTracking::SeedSpacePoint& sp) { return &sp; });

  Acts::SpacePointContainerConfig spConfig;
  spConfig.useDetailedDoubleMeasurementInfo = finderCfg.useDetailedDoubleMeasurementInfo;

  Acts::SpacePointContainerOptions spOptions;
  spOptions.beamPos = {0., 0.};

  ACTSTracking::SpacePointContainer                                       container(spacePointPtrs);
  Acts::SpacePointContainer<decltype(container), Acts::detail::RefHolder> spContainer(spConfig, spOptions, container);

  SSPointGrid grid = Acts::CylindricalSpacePointGridCreator::createGrid<SSPoint>(gridCfg, gridOpts);
  Acts::CylindricalSpacePointGridCreator::fillGrid(finderCfg, finderOpts, grid, spContainer);

  const Acts::GridBinFinder<3ul> bottomBinFinder(m_phiBottomBinLen.value(), m_zBottomBinLen.value(), 0);
  const Acts::GridBinFinder<3ul> topBinFinder(m_phiTopBinLen.value(), m_zTopBinLen.value(), 0);

  Acts::SeedFinder<SSPoint, SSPointGrid> finder(finderCfg);

  float minRange = std::numeric_limits<float>::max();
  float maxRange = std::numeric_limits<float>::lowest();
  for (const auto& coll : grid) {
    if (coll.empty())
      continue;

    const auto* firstEl = coll.front();
    const auto* lastEl  = coll.back();
    minRange            = std::min(firstEl->radius(), minRange);
    maxRange            = std::max(lastEl->radius(), maxRange);
  }

  auto spacePointsGrouping = Acts::CylindricalBinnedGroup<SSPoint>(std::move(grid), bottomBinFinder, topBinFinder);

  const Acts::Range1D<float> rMiddleSPRange(std::floor(minRange / 2) * 2 + finderCfg.deltaRMiddleMinSPRange,
                                            std::floor(maxRange / 2) * 2 - finderCfg.deltaRMiddleMaxSPRange);

  // Convert the binned group to a vector for access
  using GroupIterator = decltype(spacePointsGrouping.begin());
  using GroupValue    = std::decay_t<decltype(*std::declval<GroupIterator>())>;
  std::vector<GroupValue> spacePointGroups;
  spacePointGroups.reserve(spacePointsGrouping.grid().size());
  for (auto it = spacePointsGrouping.begin(); it != spacePointsGrouping.end(); ++it) {
    spacePointGroups.push_back(*it);
  }

  // CKF setup
  Navigator::Config navigatorCfg{trackingGeometry()};
  navigatorCfg.resolvePassive   = false;
  navigatorCfg.resolveMaterial  = true;
  navigatorCfg.resolveSensitive = true;

  Stepper    stepper(magneticField());
  Navigator  navigator(navigatorCfg);
  Propagator propagator(std::move(stepper), std::move(navigator));
  CKF        trackFinder(std::move(propagator));

  Acts::MeasurementSelector::Config measurementSelectorCfg = {
      {Acts::GeometryIdentifier(), {{}, {m_CKF_chi2CutOff}, {(std::size_t)(m_CKF_numMeasurementsCutOff)}}}};

  Acts::PropagatorPlainOptions pOptions{geometryContext(), magneticFieldContext()};
  pOptions.maxSteps = 10000;
  // Outside-in mode uses a backward first pass (from outermost seed SP inward).
  if (m_propagateBackward || m_doOutsideInCKF) {
    pOptions.direction = Acts::Direction::Backward();
  }

  std::shared_ptr<Acts::PerigeeSurface> perigeeSurface =
      Acts::Surface::makeShared<Acts::PerigeeSurface>(Acts::Vector3{0., 0., 0.});

  Acts::GainMatrixUpdater            kfUpdater;
  Acts::MeasurementSelector          measSel{measurementSelectorCfg};
  ACTSTracking::MeasurementCalibrator measCal{measurements};

  ACTSTracking::SourceLinkAccessor slAccessor;
  slAccessor.container = &sourceLinks;

  using TrackStateCreatorType = Acts::TrackStateCreator<ACTSTracking::SourceLinkAccessor::Iterator, TrackContainer>;
  TrackStateCreatorType trackStateCreator;
  trackStateCreator.sourceLinkAccessor.template connect<&ACTSTracking::SourceLinkAccessor::range>(&slAccessor);
  trackStateCreator.calibrator.template connect<&ACTSTracking::MeasurementCalibrator::calibrate>(&measCal);
  trackStateCreator.measurementSelector
      .template connect<&Acts::MeasurementSelector::select<Acts::VectorMultiTrajectory>>(&measSel);

  Acts::CombinatorialKalmanFilterExtensions<TrackContainer> extensions;
  extensions.updater.connect<&Acts::GainMatrixUpdater::operator()<Acts::VectorMultiTrajectory>>(&kfUpdater);
  extensions.createTrackStates.template connect<&TrackStateCreatorType::createTrackStates>(&trackStateCreator);

  TrackFinderOptions ckfOptions =
      TrackFinderOptions(geometryContext(), magneticFieldContext(), calibrationContext(), extensions, pOptions);

  // First pass target surface:
  //   nullptr (default) lets the CKF propagate freely from the seed and collect
  //   every reachable measurement until the navigator runs out of surfaces.
  //   Setting it makes the CKF *terminate* on that surface — not a post-CKF
  //   extrapolation hook.
  // Outside-in: first pass is a backward propagation that we want to terminate
  //   at perigee, so the smoothed parameters carry valid perigee/AtIP state.
  // Inside-out single-pass: do NOT set targetSurface even if UsePerigeeSurface
  //   is requested. With forward propagation from an inner seed, perigee sits
  //   behind the seed and the propagator aborts almost immediately, producing
  //   2-hit "tracks" (the seed only). If perigee output is needed for a
  //   single-pass forward CKF, do it as a separate post-fit extrapolation.
  if (m_doOutsideInCKF) {
    ckfOptions.targetSurface = perigeeSurface.get();
  }

  // Second-pass options for two-way CKF.
  // Inside-out (default): forward first pass → backward second pass to perigee.
  // Outside-in:           backward first pass → forward second pass outward (no target surface).
  Acts::PropagatorPlainOptions secondPOptions{geometryContext(), magneticFieldContext()};
  secondPOptions.maxSteps  = 10000;
  secondPOptions.direction = m_doOutsideInCKF ? Acts::Direction::Forward() : Acts::Direction::Backward();
  TrackFinderOptions secondOptions =
      TrackFinderOptions(geometryContext(), magneticFieldContext(), calibrationContext(), extensions, secondPOptions);
  secondOptions.skipPrePropagationUpdate = true;
  if (!m_doOutsideInCKF) {
    // Inside-out first pass: backward second pass terminates at perigee.
    secondOptions.targetSurface = perigeeSurface.get();
  }

  std::atomic<int>                     ckfActiveThreads{0};
  tbb::enumerable_thread_specific<bool> ckfThreadInit;  // default false per thread

  auto parallelSeedingAndTracking = [&](const tbb::blocked_range<size_t>& r) {
    auto& initialized = ckfThreadInit.local();
    if (!initialized) {
      initialized = true;
      info() << "CKF parallel thread #" << ++ckfActiveThreads << " started (of " << m_numThreads << " requested)"
             << endmsg;
    }
    for (size_t i = r.begin(); i != r.end(); ++i) {
      const auto& [bottom, middle, top] = spacePointGroups[i];
      // Local objects for thread safety
      std::vector<Acts::Seed<SSPoint>>        seeds;
      std::vector<Acts::BoundTrackParameters> paramseeds;
      decltype(finder)::SeedingState          state;
      state.spacePointMutableData.resize(spContainer.size());

      finder.createSeedsForGroup(finderOpts, state, spacePointsGrouping.grid(), seeds, bottom, middle, top,
                                 rMiddleSPRange);

      // Loop over seeds and get track parameters
      std::vector<Acts::Seed<ACTSTracking::SeedSpacePoint>> f_seeds;
      f_seeds.reserve(seeds.size());
      for (const Acts::Seed<SSPoint>& seed : seeds) {
        const auto& sps = seed.sp();
        f_seeds.emplace_back(*sps[0]->externalSpacePoint(), *sps[1]->externalSpacePoint(),
                             *sps[2]->externalSpacePoint());
      }

      for (const auto& seed : f_seeds) {
        // For outside-in: CKF starts from outermost (top) SP propagating backward;
        // default inside-out: start from innermost (bottom) SP propagating forward.
        const ACTSTracking::SeedSpacePoint* startSP =
            m_doOutsideInCKF ? seed.sp().back() : seed.sp().front();

        const auto&                     sourceLink = startSP->sourceLink();
        const Acts::GeometryIdentifier& geoId      = sourceLink.geometryId();
        const Acts::Surface*            surface    = trackingGeometry()->findSurface(geoId);
        if (surface == nullptr) {
          warning() << "surface with geoID " << geoId << " is not found in the tracking gemetry" << endmsg;
          continue;
        }

        // Get the magnetic field at the starting space point
        const Acts::Vector3         seedPos(startSP->x(), startSP->y(), startSP->z());
        Acts::Result<Acts::Vector3> seedField = magneticField()->getField(seedPos, magCache);
        if (!seedField.ok()) {
          throw std::runtime_error("Field lookup error: " + std::to_string(seedField.error().value()));
        }

        // estimateTrackParamsFromSeed(gctx, spRange, surface, bField) internally projects sp[0]'s
        // (bottom/inner SP) global position onto `surface` via transformFreeToBoundParameters.
        // For outside-in, `surface` is the outer SP surface and sp[0] is the inner SP — the inner
        // position projects far outside the outer sensor plane, so transformFreeToBoundParameters
        // returns an error. Fix: get the FreeVector (phi/theta/q/p correct from circle fit), replace
        // only the position with the outer SP, then project to the surface explicitly.
        Acts::Result<Acts::BoundVector> optParamsResult = [&]() -> Acts::Result<Acts::BoundVector> {
          if (m_doOutsideInCKF) {
            Acts::FreeVector freeParams = Acts::estimateTrackParamsFromSeed(seed.sp(), *seedField);
            freeParams[Acts::eFreePos0] = startSP->x();
            freeParams[Acts::eFreePos1] = startSP->y();
            freeParams[Acts::eFreePos2] = startSP->z();
            return Acts::transformFreeToBoundParameters(freeParams, *surface, geometryContext());
          }
          return Acts::estimateTrackParamsFromSeed(geometryContext(), seed.sp(), *surface, *seedField);
        }();
        if (!optParamsResult.ok()) {
          debug() << "Failed estimation of track parameters for seed." << endmsg;
          continue;
        }

        Acts::BoundVector params = *optParamsResult;

        float p = std::abs(1 / params[Acts::eBoundQOverP]);

        // --- Seed debug printout (commented out) ---
        //{
        //  const auto& sps = seed.sp();
        //  const ACTSTracking::SeedSpacePoint* botSP = sps[0];
        //  const ACTSTracking::SeedSpacePoint* midSP = sps[1];
        //  const ACTSTracking::SeedSpacePoint* topSP = sps[2];
        //  Acts::Vector3 globalParamPos =
        //      surface->localToGlobal(geometryContext(),
        //                             {params[Acts::eBoundLoc0], params[Acts::eBoundLoc1]},
        //                             {0, 0, 0});
        //  warning() << "Seed: bot geoId=" << botSP->sourceLink().geometryId()
        //            << " pos=(" << botSP->x() << "," << botSP->y() << "," << botSP->z() << ")"
        //            << " | mid geoId=" << midSP->sourceLink().geometryId()
        //            << " pos=(" << midSP->x() << "," << midSP->y() << "," << midSP->z() << ")"
        //            << " | top geoId=" << topSP->sourceLink().geometryId()
        //            << " pos=(" << topSP->x() << "," << topSP->y() << "," << topSP->z() << ")"
        //            << " start=" << (m_doOutsideInCKF ? "top" : "bot")
        //            << endmsg;
        //  warning() << "Seed params: phi=" << params[Acts::eBoundPhi]
        //            << " theta=" << params[Acts::eBoundTheta]
        //            << " p=" << p
        //            << " loc0=" << params[Acts::eBoundLoc0]
        //            << " loc1=" << params[Acts::eBoundLoc1]
        //            << " globalPos=(" << globalParamPos.transpose() << ")"
        //            << endmsg;
        //}

        // build the track covariance matrix using the smearing sigmas
        Acts::BoundMatrix cov                       = Acts::BoundMatrix::Zero();
        cov(Acts::eBoundLoc0, Acts::eBoundLoc0)     = std::pow(m_initialTrackError_pos, 2);
        cov(Acts::eBoundLoc1, Acts::eBoundLoc1)     = std::pow(m_initialTrackError_pos, 2);
        cov(Acts::eBoundTime, Acts::eBoundTime)     = std::pow(m_initialTrackError_time, 2);
        cov(Acts::eBoundPhi, Acts::eBoundPhi)       = std::pow(m_initialTrackError_phi, 2);
        cov(Acts::eBoundTheta, Acts::eBoundTheta)   = std::pow(m_initialTrackError_lambda, 2);
        cov(Acts::eBoundQOverP, Acts::eBoundQOverP) = std::pow(m_initialTrackError_relP * p / (p * p), 2);

        Acts::BoundTrackParameters paramseed(surface->getSharedPtr(), params, cov, Acts::ParticleHypothesis::pion());
        paramseeds.push_back(paramseed);

        // Compute seed state before acquiring the lock
        Acts::Vector3 globalPos =
            surface->localToGlobal(geometryContext(), {params[Acts::eBoundLoc0], params[Acts::eBoundLoc1]}, {0, 0, 0});

        Acts::Result<Acts::Vector3> hitField = magneticField()->getField(globalPos, magCache);
        if (!hitField.ok()) {
          throw std::runtime_error("Field lookup error: " + std::to_string(hitField.error().value()));
        }

        edm4hep::TrackState seedTrackState = ACTSTracking::ACTS2edm4hep_trackState(
            edm4hep::TrackState::AtFirstHit, paramseed, (*hitField)[2] / Acts::UnitConstants::T);

        // Add seed to collection, all building of seed under the lock
        {
          std::lock_guard<std::mutex> lock(m_seedMutex);
          auto                        seedTrack = seedCollection.create();
          for (const ACTSTracking::SeedSpacePoint* sp : seed.sp()) {
            seedTrack.addToTrackerHits(sp->sourceLink().edm4hepHit());
          }
          seedTrack.addToTrackStates(seedTrackState);
        }

        debug() << "Seed Paramemeters" << std::endl << paramseed << endmsg;
      }

      debug() << "Seeds found: " << std::endl << paramseeds.size() << endmsg;

      // Find the tracks
      if (!m_runCKF)
        continue;

      if (!tracking(paramseeds, trackFinder, ckfOptions, secondOptions, magCache, trackCollection).isSuccess()) {
        warning() << "Tracking failed for this event" << endmsg;
      }
    }
  };  // parallelSeedingAndTracking

  if (m_numThreads > 1) {
    arena.execute(
        [&] { tbb::parallel_for(tbb::blocked_range<size_t>(0, spacePointGroups.size()), parallelSeedingAndTracking); });
  } else {
    for (size_t i = 0; i < spacePointGroups.size(); ++i) {
      parallelSeedingAndTracking(tbb::blocked_range<size_t>(i, i + 1));
    }
  }

  auto entireEnd = std::chrono::high_resolution_clock::now();
  info() << "CKF: " << ckfActiveThreads.load() << " thread(s) active for " << spacePointGroups.size()
         << " seed groups" << endmsg;
  info() << "Track Collection Size: " << trackCollection.size() << endmsg;

  return std::make_tuple(std::move(seedCollection), std::move(trackCollection));
}

StatusCode ACTSSeededCKFTrackingAlg::tracking(const std::vector<Acts::BoundTrackParameters>& paramseeds,
                                              const CKF& trackFinder, const TrackFinderOptions& ckfOptions,
                                              const TrackFinderOptions& secondOptions,
                                              Acts::MagneticFieldProvider::Cache& magCache,
                                              edm4hep::TrackCollection&           trackCollection) const {
  debug() << "Starting CKF track finding with " << paramseeds.size() << " seeds." << endmsg;

  auto           trackContainer      = std::make_shared<Acts::VectorTrackContainer>();
  auto           trackStateContainer = std::make_shared<Acts::VectorMultiTrajectory>();
  TrackContainer tracks(trackContainer, trackStateContainer);

  for (std::size_t iseed = 0; iseed < paramseeds.size(); ++iseed) {
    tracks.clear();

    auto result = trackFinder.findTracks(paramseeds.at(iseed), ckfOptions, tracks);
    if (!result.ok()) {
      warning() << "Track fit error: " << result.error() << endmsg;
      continue;
    }

    for (const TrackContainer::TrackProxy& trackItem : result.value()) {
      auto smoothed = tracks.makeTrack();
      smoothed.copyFrom(trackItem);
      auto smoothResult = Acts::smoothTrack(geometryContext(), smoothed);
      if (!smoothResult.ok()) {
        warning() << "Track smoothing error: " << smoothResult.error() << endmsg;
        continue;
      }

      debug() << "Trajectory Summary" << endmsg;
      debug() << "\tchi2Sum       " << smoothed.chi2() << endmsg;
      debug() << "\tNDF           " << smoothed.nDoF() << endmsg;
      debug() << "\tnHoles        " << smoothed.nHoles() << endmsg;
      debug() << "\tnMeasurements " << smoothed.nMeasurements() << endmsg;
      debug() << "\tnOutliers     " << smoothed.nOutliers() << endmsg;
      debug() << "\tnStates       " << smoothed.nTrackStates() << endmsg;

      if (m_doTwoWayCKF) {
        // Two-way CKF: first pass + smooth + second pass in opposite direction.
        // Inside-out (default): forward first pass → backward to perigee.
        // Outside-in (DoOutsideInCKF):  backward first pass → forward outward into OT.
        // Ported from Athena TrackFindingAlg.cxx.
        using ConstTP  = TrackContainer::ConstTrackProxy;
        using ConstTSP = TrackContainer::ConstTrackStateProxy;

        // Find the second-pass anchor state via last-wins in trackStatesReversed():
        //   Inside-out (forward first pass): reversed goes outermost→innermost; last-wins = innermost.
        //     → second pass starts at innermost (VXD), propagates inward to perigee.
        //   Outside-in (backward first pass): reversed goes innermost→outermost; last-wins = outermost (OIT).
        //     → second pass starts at outermost (OIT), propagates outward into OT.
        // Either way: no break — always iterate all states so the HEAD-side measurement wins.
        ConstTP                 constSmoothed(smoothed);
        std::optional<ConstTSP> innermostOpt;
        for (auto st : constSmoothed.trackStatesReversed()) {
          if (!st.typeFlags().test(Acts::TrackStateFlag::MeasurementFlag)) continue;
          if (st.typeFlags().test(Acts::TrackStateFlag::OutlierFlag)) continue;
          innermostOpt = st;
        }

        if (!innermostOpt.has_value()) {
          warning() << "TwoWayCKF: no anchor measurement found, falling back to single-pass output." << endmsg;
        } else {
          const auto innermostIdx = innermostOpt->index();

          // Get smoothed parameters at innermost state; optionally inflate covariance
          // to prevent over-tight acceptance window in the second pass.
          Acts::BoundTrackParameters params2 = constSmoothed.createParametersFromState(*innermostOpt);
          if (m_inflateCovarianceTwoWay) {
            auto cov2 = *params2.covariance();
            cov2 *= m_twoWayInflateCovarianceFactor;
            params2 = Acts::BoundTrackParameters(params2.referenceSurface().getSharedPtr(),
                                                 params2.parameters(), cov2, params2.particleHypothesis());
          }

          // Second pass: from innermost measurement in opposite direction.
          // Reuse the same track container so state indices are valid for stitching.
          auto secondResult = trackFinder.findTracks(params2, secondOptions, tracks);
          if (!secondResult.ok() || secondResult.value().empty()) {
            warning() << "TwoWayCKF: second pass "
                      << (!secondResult.ok() ? std::string("FAILED: ") + secondResult.error().message()
                                             : std::string("returned EMPTY"))
                      << ", falling back to single-pass output." << endmsg;
          } else {
            auto secondTrack = tracks.makeTrack();
            secondTrack.copyFrom(*secondResult.value().begin());

            if (m_doOutsideInCKF) {
              // Outside-in stitching:
              // After copyFrom on a Forward second pass from OIT: tipIndex()=OT, stemIndex()=OITref (skip state).
              // innermostIdx = OIT measurement from the first backward pass (HEAD side of smoothed chain).
              // Link OITref.previous() = OIT_meas to continue the chain into the inner detector.
              // Full chain: OT_tip → ... → OITref → OIT_meas (1st pass) → IT → VXD (deepest stem).
              for (auto st : secondTrack.trackStates()) {
                // trackStates() iterates stem→tip (inside→out); first element = OITref (skip state at start surface)
                st.previous() = innermostIdx;
                break;
              }
              // The first backward pass has .previous() links going OUTWARD (perigee→VXD→IT→OIT).
              // After smooth, .next() links go INWARD (OIT→IT→VXD→perigee) via forwardTrackStateRange.
              // Rewire every first-pass state's .previous() to point INWARD using those .next() links,
              // so the full stitched chain traversal via .previous() goes OT→OIT→IT→VXD→perigee.
              {
                Acts::TrackIndexType prevStateIdx = Acts::kTrackIndexInvalid;
                for (auto st : smoothed.trackStates()) {
                  // smoothed.trackStates() = forwardTrackStateRange(stemIndex=OIT) → OIT, IT, VXD, perigee
                  if (prevStateIdx != Acts::kTrackIndexInvalid) {
                    tracks.trackStateContainer().getTrackState(prevStateIdx).previous() = st.index();
                  }
                  prevStateIdx = st.index();
                }
                // Last state (perigee): terminate the chain
                if (prevStateIdx != Acts::kTrackIndexInvalid) {
                  tracks.trackStateContainer().getTrackState(prevStateIdx).previous() =
                      Acts::kTrackIndexInvalid;
                }
              }
              // secondTrack.tipIndex() is already OT — no change needed.

              // Copy perigee parameters from the first backward pass (which targeted perigeeSurface)
              // so that ACTS2edm4hep_track gets valid AtIP parameters for the stitched output track.
              secondTrack.setReferenceSurface(smoothed.referenceSurface().getSharedPtr());
              secondTrack.parameters() = smoothed.parameters();
              secondTrack.covariance()  = smoothed.covariance();
            } else {
              // Inside-out + backward stitching (original two-way CKF).
              // Reverse second-pass chain.
              // Before reversal: tipIndex() = perigee (last state), chain runs perigee → ... → VXD10ref(head)
              // After reversal:  tipIndex() = VXD10ref (old head becomes new tip), chain: VXD10ref → VXD8 → ... → perigee(stem)
              // reverseTrackStates() updates tipIndex() to the old head each loop iteration.
              secondTrack.reverseTrackStates();

              // After reversal, secondTrack.tipIndex() == VXD10ref (the hole/reference state at the
              // starting surface, duplicate of the first-pass innermostIdx). We skip it and link the
              // first-pass innermost state directly to the first INNER state from the second pass (VXD8).
              // VXD10ref.previous() == VXD8's index after reversal.
              const auto firstInnerIdx = (*secondTrack.trackStatesReversed().begin()).previous();

              // Link: first-pass innermost state's previous() → first inner second-pass state
              for (auto st : smoothed.trackStatesReversed()) {
                if (st.index() == innermostIdx) {
                  st.previous() = firstInnerIdx;
                  break;
                }
              }

              // Set stitched track tip to first-pass outermost (outermost OT hit)
              secondTrack.tipIndex() = smoothed.tipIndex();
            }

            edm4hep::MutableTrack track =
                ACTSTracking::ACTS2edm4hep_track(secondTrack, magneticField(), magCache);
            std::lock_guard lock{m_trackMutex};
            trackCollection.push_back(track);
            continue;
          }
        }
      }

      // Single-pass output (DoTwoWayCKF=false, or two-way fallback)
      edm4hep::MutableTrack track = ACTSTracking::ACTS2edm4hep_track(smoothed, magneticField(), magCache);
      std::lock_guard       lock{m_trackMutex};
      trackCollection.push_back(track);
    }
  }

  return StatusCode::SUCCESS;
}
