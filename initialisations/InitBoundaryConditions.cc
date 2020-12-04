#include <math.h>  // for floor, sqrt

#include <Kokkos_Core.hpp>  // for deep_copy
#include <algorithm>        // for copy
#include <array>            // for array
#include <iostream>         // for operator<<, basic_ostream::ope...
#include <vector>           // for allocator, vector

#include "Init.h"

#include "mesh/CartesianMesh2D.h"  // for CartesianMesh2D
#include "types/MathFunctions.h"   // for max, norm, dot
#include "types/MultiArray.h"      // for operator<<
#include "utils/Utils.h"           // for indexOf

namespace initlib {

void Initialisations::initBoundaryConditions() noexcept {
  if (test->Nom == test->AdvectionX || test->Nom == test->BiAdvectionX ||
      test->Nom == test->AdvectionVitX || test->Nom == test->BiAdvectionVitX) {
    cdl->rightBC = cdl->imposedVelocity;
    cdl->rightBCValue = {{1.0, 0.0}};
    cdl->rightCellBC = cdl->periodic;
    cdl->leftBC = cdl->imposedVelocity;
    cdl->leftBCValue = {{1.0, 0.0}};
    // cdl->leftCellBC = cdl->periodic;
    cdl->topBC = cdl->symmetry;
    cdl->topBCValue = ex;
    cdl->bottomBC = cdl->symmetry;
    cdl->bottomBCValue = ex;
  }
  if (test->Nom == test->AdvectionY || test->Nom == test->BiAdvectionY) {
    cdl->bottomBC = cdl->imposedVelocity;
    cdl->bottomBCValue = {{0.0, 1.0}};
    cdl->bottomCellBC = cdl->periodic;
    cdl->topBC = cdl->imposedVelocity;
    cdl->topBCValue = {{0.0, 1.0}};
    cdl->topCellBC = cdl->periodic;

    cdl->leftBC = cdl->symmetry;
    cdl->leftBCValue = ey;

    cdl->rightBC = cdl->symmetry;
    cdl->rightBCValue = ey;
  }
  if (test->Nom == test->SodCaseX || test->Nom == test->SodCaseY ||
      test->Nom == test->BiSodCaseX || test->Nom == test->BiSodCaseY) {
    // maillage 200 5 0.005 0.02
    cdl->leftBC = cdl->symmetry;
    cdl->leftBCValue = ey;

    cdl->rightBC = cdl->symmetry;
    cdl->rightBCValue = ey;

    cdl->topBC = cdl->symmetry;
    cdl->topBCValue = ex;

    cdl->bottomBC = cdl->symmetry;
    cdl->bottomBCValue = ex;
  } else if (test->Nom == test->BiShockBubble) {
    // maillage 520 64 0.00125 0.00125
    cdl->leftBC = cdl->symmetry;
    cdl->leftBCValue = ey;

    cdl->rightBC = cdl->imposedVelocity;
    cdl->rightBCValue = {{-124.824, 0.0}};
    cdl->rightFluxBC = 1;
    cdl->rightFluxBCValue = {
        {1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -124.824, 0.0, 250000}};

    cdl->topBC = cdl->symmetry;
    cdl->topBCValue = ex;

    cdl->bottomBC = cdl->symmetry;
    cdl->bottomBCValue = ex;

  } else if (test->Nom == test->SedovTestCase ||
             test->Nom == test->BiSedovTestCase) {
    // const ℕ leftBC = symmetry; const ℝ[2] leftBCValue = ey;
    // const ℕ rightBC = imposedVelocity; const ℝ[2] rightBCValue = zeroVect;
    // const ℕ topBC = imposedVelocity; const ℝ[2] topBCValue = zeroVect;
    // const ℕ bottomBC = symmetry; const ℝ[2] bottomBCValue = ex;
    cdl->leftBC = cdl->symmetry;
    cdl->leftBCValue = ey;

    cdl->rightBC = cdl->imposedVelocity;
    cdl->rightBCValue = zeroVect;

    cdl->topBC = cdl->imposedVelocity;
    cdl->topBCValue = zeroVect;

    cdl->bottomBC = cdl->symmetry;
    cdl->bottomBCValue = ex;

    if (cstmesh->cylindrical_mesh) {
      cdl->leftBC = cdl->imposedVelocity;
      cdl->leftBCValue = zeroVect;
      cdl->rightBC = cdl->imposedVelocity;
      cdl->rightBCValue = zeroVect;
      cdl->bottomBC = cdl->symmetry;
      cdl->bottomBCValue = ex;
      cdl->topBC =  cdl->symmetry;
      cdl->topBCValue = ey;
    }
    
  } else if (test->Nom == test->TriplePoint ||
             test->Nom == test->BiTriplePoint) {
    // maillage 140 60 0.0005 0.0005
    cdl->leftBC = cdl->symmetry;
    cdl->leftBCValue = ey;

    cdl->rightBC = cdl->symmetry;
    cdl->rightBCValue = ey;

    cdl->topBC = cdl->symmetry;
    cdl->topBCValue = ex;

    cdl->bottomBC = cdl->symmetry;
    cdl->bottomBCValue = ex;
  } else if (test->Nom == test->NohTestCase ||
             test->Nom == test->BiNohTestCase) {
    // const ℕ leftBC = symmetry; const ℝ[2] leftBCValue = ey;
    // const ℕ rightBC = imposedVelocity; const ℝ[2] rightBCValue = zeroVect;
    // const ℕ topBC = imposedVelocity; const ℝ[2] topBCValue = zeroVect;
    // const ℕ bottomBC = symmetry; const ℝ[2] bottomBCValue = ex;

    cdl->leftBC = cdl->symmetry;
    cdl->leftBCValue = ey;

    cdl->rightBC = cdl->imposedVelocity;
    cdl->rightBCValue = zeroVect;

    cdl->topBC = cdl->imposedVelocity;
    cdl->topBCValue = zeroVect;

    cdl->bottomBC = cdl->symmetry;
    cdl->bottomBCValue = ex;
  } else if (test->Nom == test->UnitTestCase ||
             test->Nom == test->BiUnitTestCase) {
    cdl->rightBC = cdl->imposedVelocity;
    cdl->rightBCValue = {{1.0, 0.0}};

    cdl->leftBC = cdl->imposedVelocity;
    cdl->leftBCValue = {{1.0, 0.0}};
    cdl->leftFluxBC = 1;
    cdl->leftFluxBCValue = {
        {1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};

    cdl->bottomBC = cdl->imposedVelocity;
    cdl->bottomBCValue = {{1.0, 0.0}};

    cdl->topBC = cdl->imposedVelocity;
    cdl->topBCValue = {{1.0, 0.0}};

    // cdl->leftBC = cdl->symmetry;
    // cdl->leftBCValue = ey;

    // cdl->rightBC = cdl->symmetry;
    // cdl->rightBCValue = ey;

    // cdl->topBC = cdl->symmetry;
    // cdl->topBCValue = ex;

    // cdl->bottomBC = cdl->symmetry;
    // cdl->bottomBCValue = ex;
  } else if (test->Nom == test->Implosion || test->Nom == test->BiImplosion) {
    if (cstmesh->cylindrical_mesh == 1) {
      // rayon minimum ne bouge pas
      cdl->leftBC = cdl->imposedVelocity;
      cdl->leftBCValue = {{0.0, 0.0}};
      // rayon maximum libre
      // cdl->rightBC = cdl->freeSurface;
      cdl->rightBC = cdl->imposedVelocity;
      cdl->rightBCValue = {{0.0, 0.0}};

      // axe y
      cdl->topBC = cdl->symmetry;
      cdl->topBCValue = ey;
      // axe x
      cdl->bottomBC = cdl->symmetry;
      cdl->bottomBCValue = ex;
    }
  } else if (test->Nom == test->RiderTx || test->Nom == test->RiderTy ||
             test->Nom == test->RiderRotation || test->Nom == test->RiderT45 ||
             test->Nom == test->RiderVortex ||
             test->Nom == test->RiderDeformation ||
             test->Nom == test->RiderVortexTimeReverse ||
             test->Nom == test->RiderDeformationTimeReverse ||
             test->Nom == test->MonoRiderTx || test->Nom == test->MonoRiderTy ||
             test->Nom == test->MonoRiderRotation ||
             test->Nom == test->MonoRiderT45 ||
             test->Nom == test->MonoRiderVortex ||
             test->Nom == test->MonoRiderDeformation ||
             test->Nom == test->MonoRiderVortexTimeReverse ||
             test->Nom == test->MonoRiderDeformationTimeReverse) {
    // cdl->leftBC = cdl->freeSurface;
    // cdl->rightBC = cdl->freeSurface;
    // cdl->topBC = cdl->freeSurface;
    // cdl->bottomBC = cdl->freeSurface;
    // cdl->rightCellBC = cdl->periodic;
    // cdl->topCellBC = cdl->periodic;
  }
  cdl->FluxBC =
      cdl->leftFluxBC + cdl->rightFluxBC + cdl->bottomFluxBC + cdl->topFluxBC;
}
}  // namespace initlib
