#include <math.h>  // for sqrt

#include <Kokkos_Core.hpp>
#include <algorithm>  // for copy
#include <array>      // for array
#include <iostream>   // for operator<<, basic_ostream::operat...
#include <vector>     // for allocator, vector

#include "EucclhydRemap.h"          // for EucclhydRemap, EucclhydRemap::Opt...
#include "UtilesRemap-Impl.h"       // for EucclhydRemap::computeFluxPP
#include "mesh/CartesianMesh2D.h"   // for CartesianMesh2D
#include "types/MathFunctions.h"    // for dot
#include "types/MultiArray.h"       // for operator<<
#include "utils/Utils.h"            // for indexOf

/**
 * Job computeGradPhiFace2 called @12.0 in executeTimeLoopN method.
 * In variables: Uremap1, deltaxLagrange, projectionOrder, vLagrange, x_then_y_n
 * Out variables: gradPhiFace2
 */
void EucclhydRemap::computeGradPhiFace2() noexcept {
  if (options->projectionOrder > 1) {
    if (x_then_y_n) {
      auto innerHorizontalFaces(mesh->getInnerHorizontalFaces());
      int nbInnerHorizontalFaces(mesh->getNbInnerHorizontalFaces());
      Kokkos::parallel_for(
          "computeGradPhiFace2", nbInnerHorizontalFaces,
          KOKKOS_LAMBDA(const int& fInnerHorizontalFaces) {
            size_t fId(innerHorizontalFaces[fInnerHorizontalFaces]);
            int fFaces(utils::indexOf(mesh->getFaces(), fId));
            int cfFrontCellF(mesh->getFrontCell(fId));
            int cfId(cfFrontCellF);
            int cfCells(cfId);
            int cbBackCellF(mesh->getBackCell(fId));
            int cbId(cbBackCellF);
            int cbCells(cbId);

            gradPhiFace2(fFaces) = (Phi(cfCells) - Phi(cbCells)) /
                deltaxLagrange(fFaces);

            int n1FirstNodeOfFaceF(mesh->getFirstNodeOfFace(fId));
            int n1Id(n1FirstNodeOfFaceF);
            int n1Nodes(n1Id);
            int n2SecondNodeOfFaceF(mesh->getSecondNodeOfFace(fId));
            int n2Id(n2SecondNodeOfFaceF);
            int n2Nodes(n2Id);
            LfLagrange(fFaces) =
                sqrt((XLagrange(n1Nodes)[0] - XLagrange(n2Nodes)[0]) *
                         (XLagrange(n1Nodes)[0] - XLagrange(n2Nodes)[0]) +
                     (XLagrange(n1Nodes)[1] - XLagrange(n2Nodes)[1]) *
                         (XLagrange(n1Nodes)[1] - XLagrange(n2Nodes)[1]));
          });
      Kokkos::parallel_for(
          "computeGradPhiFace2", nbCells, KOKKOS_LAMBDA(const int& cCells) {
            int cId(cCells);
            // seconde methode
            HvLagrange(cCells) = 0.;
            int frRightFaceOfCellC(mesh->getRightFaceOfCell(cId));
            size_t frId(frRightFaceOfCellC);
            int frFaces(utils::indexOf(mesh->getFaces(), frId));
            HvLagrange(cCells) += 0.5 * faceLengthLagrange(frFaces);

            int flLeftFaceOfCellC(mesh->getLeftFaceOfCell(cId));
            size_t flId(flLeftFaceOfCellC);
            int flFaces(utils::indexOf(mesh->getFaces(), flId));
            HvLagrange(cCells) += 0.5 * faceLengthLagrange(flFaces);
          });
    } else {
      auto innerVerticalFaces(mesh->getInnerVerticalFaces());
      int nbInnerVerticalFaces(mesh->getNbInnerVerticalFaces());
      Kokkos::parallel_for(
          "computeGradPhiFace2", nbInnerVerticalFaces,
          KOKKOS_LAMBDA(const int& fInnerVerticalFaces) {
            size_t fId(innerVerticalFaces[fInnerVerticalFaces]);
            int fFaces(utils::indexOf(mesh->getFaces(), fId));
            int cfFrontCellF(mesh->getFrontCell(fId));
            int cfId(cfFrontCellF);
            int cfCells(cfId);
            int cbBackCellF(mesh->getBackCell(fId));
            int cbId(cbBackCellF);
            int cbCells(cbId);
            //
            gradPhiFace2(fFaces) = (Phi(cfCells) - Phi(cbCells)) /
                deltaxLagrange(fFaces);
            //
            int n1FirstNodeOfFaceF(mesh->getFirstNodeOfFace(fId));
            int n1Id(n1FirstNodeOfFaceF);
            int n1Nodes(n1Id);
            int n2SecondNodeOfFaceF(mesh->getSecondNodeOfFace(fId));
            int n2Id(n2SecondNodeOfFaceF);
            int n2Nodes(n2Id);
            LfLagrange(fFaces) =
                sqrt((XLagrange(n1Nodes)[0] - XLagrange(n2Nodes)[0]) *
                         (XLagrange(n1Nodes)[0] - XLagrange(n2Nodes)[0]) +
                     (XLagrange(n1Nodes)[1] - XLagrange(n2Nodes)[1]) *
                         (XLagrange(n1Nodes)[1] - XLagrange(n2Nodes)[1]));
          });
      Kokkos::parallel_for(
          "computeGradPhiFace2", nbCells, KOKKOS_LAMBDA(const int& cCells) {
            int cId(cCells);
            // seconde methode
            HvLagrange(cCells) = 0.;
            int fbBottomFaceOfCellC(mesh->getBottomFaceOfCell(cId));
            size_t fbId(fbBottomFaceOfCellC);
            int fbFaces(utils::indexOf(mesh->getFaces(), fbId));
            HvLagrange(cCells) += 0.5 * faceLengthLagrange(fbFaces);

            int ftTopFaceOfCellC(mesh->getTopFaceOfCell(cId));
            size_t ftId(ftTopFaceOfCellC);
            int ftFaces(utils::indexOf(mesh->getFaces(), ftId));
            HvLagrange(cCells) += 0.5 * faceLengthLagrange(ftFaces);
          });
    }
  }
}

/**
 * Job computeGradPhi2 called @13.0 in executeTimeLoopN method.
 * In variables: gradPhiFace2, projectionLimiterId, projectionOrder, x_then_y_n
 * Out variables: gradPhi2
 */
void EucclhydRemap::computeGradPhi2() noexcept {
  if (options->projectionOrder > 1) {
    if (x_then_y_n) {
      // std::cout << " Phase 2 Verticale computeGradPhi2 " << std::endl;
      Kokkos::parallel_for(
          "computeGradPhi2", nbCells, KOKKOS_LAMBDA(const int& cCells) {
            int cId(cCells);
            int fbBottomFaceOfCellC(mesh->getBottomFaceOfCell(cId));
            size_t fbId(fbBottomFaceOfCellC);
            int fbFaces(utils::indexOf(mesh->getFaces(), fbId));
            int ftTopFaceOfCellC(mesh->getTopFaceOfCell(cId));
            size_t ftId(ftTopFaceOfCellC);
            int ftFaces(utils::indexOf(mesh->getFaces(), ftId));
            // maille dessous
            int cfFrontCellF(mesh->getFrontCell(fbId));
            int cfId(cfFrontCellF);
            int cfCells(cfId);
            int fbFacesOfCellC(utils::indexOf(mesh->getFacesOfCell(cId), fbId));
            // maille dessus
            int cbBackCellF(mesh->getBackCell(ftId));
            int cbId(cbBackCellF);
            int cbCells(cbId);
            int ftFacesOfCellC(utils::indexOf(mesh->getFacesOfCell(cId), ftId));

            if (cbCells == -1) cbCells = cCells;
            if (cfCells == -1) cfCells = cCells;
            bool voisinage_pure =
                (limiteurs->projectionLimiteurMixte == 1) &&
                (mixte(cCells) == 0 && mixte(cfCells) == 0 &&
                 mixte(cbCells) == 0 && pure(cCells) == pure(cfCells) &&
                 pure(cCells) == pure(cbCells));

            int limiter = limiteurs->projectionLimiterId;
            if ((limiteurs->projectionAvecPlateauPente == 1) && voisinage_pure)
              limiter = limiteurs->projectionLimiterIdPure;

            gradPhi2(cCells) = computeAndLimitGradPhi(
                limiter, gradPhiFace2(fbFaces), gradPhiFace2(ftFaces),
                Phi(cCells), Phi(cbCells), Phi(cfCells), HvLagrange(cCells),
                HvLagrange(cbCells), HvLagrange(cfCells));

            if (limiteurs->projectionAvecPlateauPente == 1) {
              RealArray1D<dim> exy = {{0.0, 1.0}};

              double Flux_sortant_av =
                  dot(outerFaceNormal(cCells, fbFacesOfCellC),
                                     exy) *
                  faceNormalVelocity(fbFaces);
              if (voisinage_pure)
                deltaPhiFaceAr(cCells) = computeFluxPPPure(
                    gradPhi2(cCells), Phi(cCells), Phi(cbCells), Phi(cfCells),
                    HvLagrange(cCells), HvLagrange(cbCells),
                    HvLagrange(cfCells), Flux_sortant_av, gt->deltat_n, 0,
                    cCells, options->threshold);
              else
                deltaPhiFaceAr(cCells) = computeFluxPP(
                    gradPhi2(cCells), Phi(cCells), Phi(cbCells), Phi(cfCells),
                    HvLagrange(cCells), HvLagrange(cbCells),
                    HvLagrange(cfCells), Flux_sortant_av, gt->deltat_n, 0,
                    cCells, options->threshold);

              double Flux_sortant_ar =
                  dot(outerFaceNormal(cCells, ftFacesOfCellC),
                                     exy) *
                  faceNormalVelocity(ftFaces);
              if (voisinage_pure)
                deltaPhiFaceAv(cCells) = computeFluxPPPure(
                    gradPhi2(cCells), Phi(cCells), Phi(cbCells), Phi(cfCells),
                    HvLagrange(cCells), HvLagrange(cbCells),
                    HvLagrange(cfCells), Flux_sortant_ar, gt->deltat_n, 1,
                    cCells, options->threshold);
              else
                deltaPhiFaceAv(cCells) = computeFluxPP(
                    gradPhi2(cCells), Phi(cCells), Phi(cbCells), Phi(cfCells),
                    HvLagrange(cCells), HvLagrange(cbCells),
                    HvLagrange(cfCells), Flux_sortant_ar, gt->deltat_n, 1,
                    cCells, options->threshold);
            }
          });
    } else {
      // std::cout << " Phase 2 Horizontale computeGradPhi2 " << std::endl;
      Kokkos::parallel_for(
          "computeGradPhi2", nbCells, KOKKOS_LAMBDA(const int& cCells) {
            int cId(cCells);
            int frRightFaceOfCellC(mesh->getRightFaceOfCell(cId));
            size_t frId(frRightFaceOfCellC);
            int frFaces(utils::indexOf(mesh->getFaces(), frId));
            int flLeftFaceOfCellC(mesh->getLeftFaceOfCell(cId));
            size_t flId(flLeftFaceOfCellC);
            int flFaces(utils::indexOf(mesh->getFaces(), flId));

            // maille devant
            int cfFrontCellF(mesh->getFrontCell(frId));
            int cfId(cfFrontCellF);
            int cfCells(cfId);
            int frFacesOfCellC(utils::indexOf(mesh->getFacesOfCell(cId), frId));
            // maille deriere
            int cbBackCellF(mesh->getBackCell(flId));
            int cbId(cbBackCellF);
            int cbCells(cbId);
            int flFacesOfCellC(utils::indexOf(mesh->getFacesOfCell(cId), flId));

            if (cbCells == -1) cbCells = cCells;
            if (cfCells == -1) cfCells = cCells;
            bool voisinage_pure =
                (limiteurs->projectionLimiteurMixte == 1) &&
                (mixte(cCells) == 0 && mixte(cfCells) == 0 &&
                 mixte(cbCells) == 0 && pure(cCells) == pure(cfCells) &&
                 pure(cCells) == pure(cbCells));

            int limiter = limiteurs->projectionLimiterId;
            if ((limiteurs->projectionAvecPlateauPente == 1) && voisinage_pure)
              limiter = limiteurs->projectionLimiterIdPure;
            //
            gradPhi2(cCells) = computeAndLimitGradPhi(
                limiter, gradPhiFace2(frFaces), gradPhiFace2(flFaces),
                Phi(cCells), Phi(cfCells), Phi(cbCells), HvLagrange(cCells),
                HvLagrange(cfCells), HvLagrange(cbCells));
            //
            if (limiteurs->projectionAvecPlateauPente == 1) {
              RealArray1D<dim> exy = {{1.0, 0.0}};
              double Flux_sortant_ar =
                  dot(outerFaceNormal(cCells, flFacesOfCellC),
                                     exy) *
                  faceNormalVelocity(flFaces);
              if (voisinage_pure)
                deltaPhiFaceAr(cCells) = computeFluxPPPure(
                    gradPhi2(cCells), Phi(cCells), Phi(cfCells), Phi(cbCells),
                    HvLagrange(cCells), HvLagrange(cfCells),
                    HvLagrange(cbCells), Flux_sortant_ar, gt->deltat_n, 0,
                    cCells, options->threshold);
              else
                deltaPhiFaceAr(cCells) = computeFluxPP(
                    gradPhi2(cCells), Phi(cCells), Phi(cfCells), Phi(cbCells),
                    HvLagrange(cCells), HvLagrange(cfCells),
                    HvLagrange(cbCells), Flux_sortant_ar, gt->deltat_n, 0,
                    cCells, options->threshold);

              double Flux_sortant_av =
                  dot(outerFaceNormal(cCells, frFacesOfCellC),
                                     exy) *
                  faceNormalVelocity(frFaces);
              if (voisinage_pure)
                deltaPhiFaceAv(cCells) = computeFluxPPPure(
                    gradPhi2(cCells), Phi(cCells), Phi(cfCells), Phi(cbCells),
                    HvLagrange(cCells), HvLagrange(cfCells),
                    HvLagrange(cbCells), Flux_sortant_av, gt->deltat_n, 1,
                    cCells, options->threshold);
              else
                deltaPhiFaceAv(cCells) = computeFluxPP(
                    gradPhi2(cCells), Phi(cCells), Phi(cfCells), Phi(cbCells),
                    HvLagrange(cCells), HvLagrange(cfCells),
                    HvLagrange(cbCells), Flux_sortant_av, gt->deltat_n, 1,
                    cCells, options->threshold);
            }
          });
    }
  }
}

/**
 * Job computeUpwindFaceQuantitiesForProjection2 called @14.0 in
 * executeTimeLoopN method. In variables: Uremap1, XcLagrange, Xf,
 * deltaxLagrange, faceNormal, faceNormalVelocity, gradPhi2, vLagrange,
 * x_then_y_n Out variables: phiFace2
 */
void EucclhydRemap::computeUpwindFaceQuantitiesForProjection2() noexcept {
  if (x_then_y_n) {
    // std::cout << " Phase Projection 2 Verticale " << std::endl;
    auto innerHorizontalFaces(mesh->getInnerHorizontalFaces());
    int nbInnerHorizontalFaces(mesh->getNbInnerHorizontalFaces());
    Kokkos::parallel_for(
        "computeUpwindFaceQuantitiesForProjection2", nbInnerHorizontalFaces,
        KOKKOS_LAMBDA(const int& fInnerHorizontalFaces) {
          size_t fId(innerHorizontalFaces[fInnerHorizontalFaces]);
          int fFaces(utils::indexOf(mesh->getFaces(), fId));
          int cfFrontCellF(mesh->getFrontCell(fId));
          int cfId(cfFrontCellF);
          int cfCells(cfId);
          int cbBackCellF(mesh->getBackCell(fId));
          int cbId(cbBackCellF);
          int cbCells(cbId);
          // phiFace2 correspond
          // à la valeur de phi(x) à la face pour l'ordre 2 sans plateau pente
          // ou l'ordre 3 à la valeur du flux (integration de phi(x)) pour
          // l'ordre 2 avec Plateau-Pente
          if (options->projectionOrder == 2) {
            if (limiteurs->projectionAvecPlateauPente == 0) {
              phiFace2(fFaces) = computeUpwindFaceQuantities(
                  faceNormal(fFaces), faceNormalVelocity(fFaces),
                  deltaxLagrange(fFaces), Xf(fFaces),
                  (Uremap1(cbCells) / vLagrange(cbCells)),
                  gradPhi2(cbCells), XcLagrange(cbCells),
                  (Uremap1(cfCells) / vLagrange(cfCells)),
                  gradPhi2(cfCells), XcLagrange(cfCells));
            } else {
              phiFace2(fFaces) = 
                  deltaPhiFaceAv(cfCells) - deltaPhiFaceAr(cbCells);
            }
          } else if (options->projectionOrder == 3) {
            // cfCells est en dessous de cCells
            // cffCells est en dessous de cfcells
            // cfffCells est en dessous de cffcells
            int cffCells(getBottomCells(cfCells));
            int cfffCells(getBottomCells(cffCells));

            // cbCells est au dessus de cCells
            // cbbCells est au dessus de cbcells
            // cbbbCells est au dessus de cbbcells
            int cbbCells(getTopCells(cbCells));
            int cbbbCells(getTopCells(cbbCells));

            phiFace2(fFaces) = computeVecFluxOrdre3(
                (Uremap1(cfffCells) / vLagrange(cfffCells)),
                (Uremap1(cffCells)  / vLagrange(cffCells)),
                (Uremap1(cfCells)   / vLagrange(cfCells)),
                (Uremap1(cbCells)   / vLagrange(cbCells)),
                (Uremap1(cbbCells)  / vLagrange(cbbCells)),
                (Uremap1(cbbbCells) / vLagrange(cbbbCells)),
                HvLagrange(cfffCells), HvLagrange(cffCells),
                HvLagrange(cfCells), HvLagrange(cbCells), HvLagrange(cbbCells),
                HvLagrange(cbbbCells), faceNormalVelocity(fFaces),
                gt->deltat_n);
          }
          // std::cout << " face " << fFaces << " " << cfCells << "  " <<
          // cbCells << " flux " << phiFace2(fFaces) << std::endl;
        });
  } else {
    // std::cout << " Phase Projection 2 Horizontale " << std::endl;
    auto innerVerticalFaces(mesh->getInnerVerticalFaces());
    int nbInnerVerticalFaces(mesh->getNbInnerVerticalFaces());
    Kokkos::parallel_for(
        "computeUpwindFaceQuantitiesForProjection2", nbInnerVerticalFaces,
        KOKKOS_LAMBDA(const int& fInnerVerticalFaces) {
          size_t fId(innerVerticalFaces[fInnerVerticalFaces]);
          int fFaces(utils::indexOf(mesh->getFaces(), fId));
          int cfFrontCellF(mesh->getFrontCell(fId));
          int cfId(cfFrontCellF);
          int cfCells(cfId);
          int cbBackCellF(mesh->getBackCell(fId));
          int cbId(cbBackCellF);
          int cbCells(cbId);
          // phiFace1 correspond
          // à la valeur de phi(x) à la face pour l'ordre 2 sans plateau pente
          // ou l'ordre 3 à la valeur du flux (integration de phi(x)) pour
          // l'ordre 2 avec Plateau-Pente
          if (options->projectionOrder == 2) {
            if (limiteurs->projectionAvecPlateauPente == 0) {
              phiFace2(fFaces) = computeUpwindFaceQuantities(
                  faceNormal(fFaces), faceNormalVelocity(fFaces),
                  deltaxLagrange(fFaces), Xf(fFaces),
                  (Uremap1(cbCells) / vLagrange(cbCells)),
                  gradPhi2(cbCells), XcLagrange(cbCells),
                  (Uremap1(cfCells) / vLagrange(cfCells)),
                  gradPhi2(cfCells), XcLagrange(cfCells));
            } else {
              phiFace2(fFaces) = 
                  deltaPhiFaceAv(cbCells) - deltaPhiFaceAr(cfCells);
            }
          } else if (options->projectionOrder == 3) {
            // cfCells est à droite de cCells
            // cffcells est à droite de cfcells
            // cffcells est à droite de cffcells
            int cffCells(getRightCells(cfCells));
            int cfffCells(getRightCells(cffCells));

            // cbCells est à gauche de cCells
            // cbbcells est à gauche de cbcells
            // cbbbcells est à gauche de cbbcells
            int cbbCells(getLeftCells(cbCells));
            int cbbbCells(getLeftCells(cbbCells));
            phiFace2(fFaces) = computeVecFluxOrdre3(
                (Uremap1(cbbbCells) / vLagrange(cbbbCells)),
                (Uremap1(cbbCells)  / vLagrange(cbbCells)),
                (Uremap1(cbCells)   / vLagrange(cbCells)),
                (Uremap1(cfCells)   / vLagrange(cfCells)),
                (Uremap1(cffCells)  / vLagrange(cffCells)),
                (Uremap1(cfffCells) / vLagrange(cfffCells)),
                HvLagrange(cbbbCells), HvLagrange(cbbCells),
                HvLagrange(cbCells), HvLagrange(cfCells), HvLagrange(cffCells),
                HvLagrange(cfffCells), faceNormalVelocity(fFaces),
                gt->deltat_n);
          }
          // std::cout << " face " << fFaces << " " << cfCells << "  " <<
          // cbCells << " flux " << phiFace2(fFaces) << std::endl;
        });
  }
}

/**
 * Job computeUremap2 called @15.0 in executeTimeLoopN method.
 * In variables: Uremap1, deltat_n, faceLength, faceNormal, faceNormalVelocity,
 * outerFaceNormal, phiFace2, x_then_y_n Out variables: Uremap2
 */
void EucclhydRemap::computeUremap2() noexcept {
  RealArray1D<dim> exy = xThenYToDirection(!x_then_y_n);
  Kokkos::parallel_for(
      "computeUremap2", nbCells, KOKKOS_LAMBDA(const int& cCells) {
        int cId(cCells);
        RealArray1D<nbequamax> reduction9 = Uzero;
        {
          auto neighbourCellsC(mesh->getNeighbourCells(cId));
          for (int dNeighbourCellsC = 0;
               dNeighbourCellsC < neighbourCellsC.size(); dNeighbourCellsC++) {
            int dId(neighbourCellsC[dNeighbourCellsC]);
            int fCommonFaceCD(mesh->getCommonFace(cId, dId));
            size_t fId(fCommonFaceCD);
            int fFaces(utils::indexOf(mesh->getFaces(), fId));
            int fFacesOfCellC(utils::indexOf(mesh->getFacesOfCell(cId), fId));
            reduction9 = 
                reduction9 + (computeRemapFlux(
                                options->projectionOrder,
                                limiteurs->projectionAvecPlateauPente,
                                faceNormalVelocity(fFaces), faceNormal(fFaces),
                                faceLength(fFaces), phiFace2(fFaces),
                                outerFaceNormal(cCells, fFacesOfCellC), exy,
                                gt->deltat_n));
            //

            //
          }
          if (cdl->FluxBC > 0) {
            // flux exterieur

            reduction9 = 
                reduction9 + (computeBoundaryFluxes(2, cCells, exy));
          }
        }

        Uremap2(cCells) = Uremap1(cCells) - reduction9;
      });
}
