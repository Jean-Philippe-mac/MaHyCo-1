#include <math.h>  // for sqrt

#include <Kokkos_Core.hpp>
#include <algorithm>  // for copy
#include <array>      // for array
#include <iostream>   // for operator<<, basic_ostream::operat...
#include <vector>     // for allocator, vector

#include "Remap.h"                 // for Remap, Remap::Opt...
#include "UtilesRemap-Impl.h"      // for Remap::computeFluxPP
#include "mesh/CartesianMesh2D.h"  // for CartesianMesh2D
#include "types/MathFunctions.h"   // for dot
#include "types/MultiArray.h"      // for operator<<
#include "utils/Utils.h"           // for indexOf

/**
 *******************************************************************************
 * \file computeGradPhiFace2()
 * \brief phase 2 de projection : premiere etapes
 *  calcul des gradients aux faces verticales ou horizontales suivant le cas
 *  calcul des longueurs des faces verticales ou horizontales suivant le cas
 *  calcul des largeurs de cellule dans le sens verticales ou horizontales
 *suivant le cas \param \return gradPhiFace2, LfLagrange, HvLagrange
 *******************************************************************************
 */
void Remap::computeGradPhiFace2() noexcept {
  if (options->projectionOrder > 1) {
    if (varlp->x_then_y_n) {
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

            gradPhiFace2(fFaces) = (varlp->Phi(cfCells) - varlp->Phi(cbCells)) /
                                   varlp->deltaxLagrange(fFaces);

            int n1FirstNodeOfFaceF(mesh->getFirstNodeOfFace(fId));
            int n1Id(n1FirstNodeOfFaceF);
            int n1Nodes(n1Id);
            int n2SecondNodeOfFaceF(mesh->getSecondNodeOfFace(fId));
            int n2Id(n2SecondNodeOfFaceF);
            int n2Nodes(n2Id);
            LfLagrange(fFaces) = sqrt(
                (varlp->XLagrange(n1Nodes)[0] - varlp->XLagrange(n2Nodes)[0]) *
                    (varlp->XLagrange(n1Nodes)[0] -
                     varlp->XLagrange(n2Nodes)[0]) +
                (varlp->XLagrange(n1Nodes)[1] - varlp->XLagrange(n2Nodes)[1]) *
                    (varlp->XLagrange(n1Nodes)[1] -
                     varlp->XLagrange(n2Nodes)[1]));
          });
      Kokkos::parallel_for(
          "computeGradPhiFace2", nbCells, KOKKOS_LAMBDA(const int& cCells) {
            int cId(cCells);
            // seconde methode
            HvLagrange(cCells) = 0.;
            int frRightFaceOfCellC(mesh->getRightFaceOfCell(cId));
            size_t frId(frRightFaceOfCellC);
            int frFaces(utils::indexOf(mesh->getFaces(), frId));
            HvLagrange(cCells) += 0.5 * varlp->faceLengthLagrange(frFaces);

            int flLeftFaceOfCellC(mesh->getLeftFaceOfCell(cId));
            size_t flId(flLeftFaceOfCellC);
            int flFaces(utils::indexOf(mesh->getFaces(), flId));
            HvLagrange(cCells) += 0.5 * varlp->faceLengthLagrange(flFaces);
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
            gradPhiFace2(fFaces) = (varlp->Phi(cfCells) - varlp->Phi(cbCells)) /
                                   varlp->deltaxLagrange(fFaces);
            //
            int n1FirstNodeOfFaceF(mesh->getFirstNodeOfFace(fId));
            int n1Id(n1FirstNodeOfFaceF);
            int n1Nodes(n1Id);
            int n2SecondNodeOfFaceF(mesh->getSecondNodeOfFace(fId));
            int n2Id(n2SecondNodeOfFaceF);
            int n2Nodes(n2Id);
            LfLagrange(fFaces) = sqrt(
                (varlp->XLagrange(n1Nodes)[0] - varlp->XLagrange(n2Nodes)[0]) *
                    (varlp->XLagrange(n1Nodes)[0] -
                     varlp->XLagrange(n2Nodes)[0]) +
                (varlp->XLagrange(n1Nodes)[1] - varlp->XLagrange(n2Nodes)[1]) *
                    (varlp->XLagrange(n1Nodes)[1] -
                     varlp->XLagrange(n2Nodes)[1]));
          });
      Kokkos::parallel_for(
          "computeGradPhiFace2", nbCells, KOKKOS_LAMBDA(const int& cCells) {
            int cId(cCells);
            // seconde methode
            HvLagrange(cCells) = 0.;
            int fbBottomFaceOfCellC(mesh->getBottomFaceOfCell(cId));
            size_t fbId(fbBottomFaceOfCellC);
            int fbFaces(utils::indexOf(mesh->getFaces(), fbId));
            HvLagrange(cCells) += 0.5 * varlp->faceLengthLagrange(fbFaces);

            int ftTopFaceOfCellC(mesh->getTopFaceOfCell(cId));
            size_t ftId(ftTopFaceOfCellC);
            int ftFaces(utils::indexOf(mesh->getFaces(), ftId));
            HvLagrange(cCells) += 0.5 * varlp->faceLengthLagrange(ftFaces);
          });
    }
  }
}
/**
 *******************************************************************************
 * \file computeGradPhi2()
 * \brief phase 2 de projection : seconde etape
 *        calcul du gradient aux mailles limites
 *        calcul des flux pente-borne arriere ou avant aux mailles
 *                 à partir du gradient precedent
 * \param
 * \return gradPhi2, deltaPhiFaceAr, deltaPhiFaceAv
 *******************************************************************************
 */
void Remap::computeGradPhi2() noexcept {
  if (options->projectionOrder > 1) {
    if (varlp->x_then_y_n) {
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
            RealArray1D<dim> exy = {{0.0, 1.0}};

            if (cbCells == -1) cbCells = cCells;
            if (cfCells == -1) cfCells = cCells;
            bool voisinage_pure =
                (limiteurs->projectionLimiteurMixte == 1) &&
                (varlp->mixte(cCells) == 0 && varlp->mixte(cfCells) == 0 &&
                 varlp->mixte(cbCells) == 0 &&
                 varlp->pure(cCells) == varlp->pure(cfCells) &&
                 varlp->pure(cCells) == varlp->pure(cbCells));

            int limiter = limiteurs->projectionLimiterId;
            if ((limiteurs->projectionAvecPlateauPente == 1) && voisinage_pure)
              limiter = limiteurs->projectionLimiterIdPure;

            gradPhi2(cCells) = computeAndLimitGradPhi(
                limiter, gradPhiFace2(fbFaces), gradPhiFace2(ftFaces),
                varlp->Phi(cCells), varlp->Phi(cbCells), varlp->Phi(cfCells),
                HvLagrange(cCells), HvLagrange(cbCells), HvLagrange(cfCells));

            if (limiteurs->projectionAvecPlateauPente == 1) {
              if (cstmesh->cylindrical_mesh) exy = varlp->faceNormal(fbFaces);

              double Flux_sortant_av =
                  dot(varlp->outerFaceNormal(cCells, fbFacesOfCellC), exy) *
                  varlp->faceNormalVelocity(fbFaces);

              if (cstmesh->cylindrical_mesh) exy = varlp->faceNormal(ftFaces);

              double Flux_sortant_ar =
                  dot(varlp->outerFaceNormal(cCells, ftFacesOfCellC), exy) *
                  varlp->faceNormalVelocity(ftFaces);

              double flux_dual = 0.5 * (varlp->faceNormalVelocity(fbFaces) +
                                        varlp->faceNormalVelocity(ftFaces));
              int calcul_flux_dual(0);
              if (options->methode_flux_masse == 2) calcul_flux_dual = 1;

              if (voisinage_pure)
                computeFluxPPPure(
                    gradPhi2(cCells), varlp->Phi(cCells), varlp->Phi(cbCells),
                    varlp->Phi(cfCells), HvLagrange(cCells),
                    HvLagrange(cbCells), HvLagrange(cfCells), Flux_sortant_av,
                    gt->deltat_n, 0, cCells, options->threshold,
                    limiteurs->projectionPlateauPenteComplet, flux_dual,
                    calcul_flux_dual, &deltaPhiFaceAr(cCells),
                    &DualphiFlux2(cCells));
              else
                computeFluxPP(gradPhi2(cCells), varlp->Phi(cCells),
                              varlp->Phi(cbCells), varlp->Phi(cfCells),
                              HvLagrange(cCells), HvLagrange(cbCells),
                              HvLagrange(cfCells), Flux_sortant_av,
                              gt->deltat_n, 0, cCells, options->threshold,
                              limiteurs->projectionPlateauPenteComplet,
                              flux_dual, calcul_flux_dual,
                              &deltaPhiFaceAr(cCells), &DualphiFlux2(cCells));

              // pour avoir un flux dual 2D
              DualphiFlux2(cCells) *=
                  (varlp->faceLength(fbFaces) + varlp->faceLength(ftFaces)) *
                  0.5;

              if (voisinage_pure)
                computeFluxPPPure(
                    gradPhi2(cCells), varlp->Phi(cCells), varlp->Phi(cbCells),
                    varlp->Phi(cfCells), HvLagrange(cCells),
                    HvLagrange(cbCells), HvLagrange(cfCells), Flux_sortant_ar,
                    gt->deltat_n, 1, cCells, options->threshold,
                    limiteurs->projectionPlateauPenteComplet, flux_dual,
                    calcul_flux_dual, &deltaPhiFaceAv(cCells), &Bidon2(cCells));
              else
                computeFluxPP(
                    gradPhi2(cCells), varlp->Phi(cCells), varlp->Phi(cbCells),
                    varlp->Phi(cfCells), HvLagrange(cCells),
                    HvLagrange(cbCells), HvLagrange(cfCells), Flux_sortant_ar,
                    gt->deltat_n, 1, cCells, options->threshold,
                    limiteurs->projectionPlateauPenteComplet, flux_dual,
                    calcul_flux_dual, &deltaPhiFaceAv(cCells), &Bidon2(cCells));
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
            RealArray1D<dim> exy = {{1.0, 0.0}};

            if (cbCells == -1) cbCells = cCells;
            if (cfCells == -1) cfCells = cCells;
            bool voisinage_pure =
                (limiteurs->projectionLimiteurMixte == 1) &&
                (varlp->mixte(cCells) == 0 && varlp->mixte(cfCells) == 0 &&
                 varlp->mixte(cbCells) == 0 &&
                 varlp->pure(cCells) == varlp->pure(cfCells) &&
                 varlp->pure(cCells) == varlp->pure(cbCells));

            int limiter = limiteurs->projectionLimiterId;
            if ((limiteurs->projectionAvecPlateauPente == 1) && voisinage_pure)
              limiter = limiteurs->projectionLimiterIdPure;
            //
            gradPhi2(cCells) = computeAndLimitGradPhi(
                limiter, gradPhiFace2(frFaces), gradPhiFace2(flFaces),
                varlp->Phi(cCells), varlp->Phi(cfCells), varlp->Phi(cbCells),
                HvLagrange(cCells), HvLagrange(cfCells), HvLagrange(cbCells));
            //
            if (limiteurs->projectionAvecPlateauPente == 1) {
              if (cstmesh->cylindrical_mesh) exy = varlp->faceNormal(flFaces);

              double Flux_sortant_ar =
                  dot(varlp->outerFaceNormal(cCells, flFacesOfCellC), exy) *
                  varlp->faceNormalVelocity(flFaces);

              if (cstmesh->cylindrical_mesh) exy = varlp->faceNormal(frFaces);

              double Flux_sortant_av =
                  dot(varlp->outerFaceNormal(cCells, frFacesOfCellC), exy) *
                  varlp->faceNormalVelocity(frFaces);

              double flux_dual = 0.5 * (varlp->faceNormalVelocity(flFaces) +
                                        varlp->faceNormalVelocity(frFaces));
              int calcul_flux_dual(0);
              if (options->methode_flux_masse == 2) calcul_flux_dual = 1;

              if (voisinage_pure)
                computeFluxPPPure(
                    gradPhi2(cCells), varlp->Phi(cCells), varlp->Phi(cfCells),
                    varlp->Phi(cbCells), HvLagrange(cCells),
                    HvLagrange(cfCells), HvLagrange(cbCells), Flux_sortant_ar,
                    gt->deltat_n, 0, cCells, options->threshold,
                    limiteurs->projectionPlateauPenteComplet, flux_dual,
                    calcul_flux_dual, &deltaPhiFaceAr(cCells),
                    &DualphiFlux2(cCells));
              else
                computeFluxPP(gradPhi2(cCells), varlp->Phi(cCells),
                              varlp->Phi(cfCells), varlp->Phi(cbCells),
                              HvLagrange(cCells), HvLagrange(cfCells),
                              HvLagrange(cbCells), Flux_sortant_ar,
                              gt->deltat_n, 0, cCells, options->threshold,
                              limiteurs->projectionPlateauPenteComplet,
                              flux_dual, calcul_flux_dual,
                              &deltaPhiFaceAr(cCells), &DualphiFlux2(cCells));

              // pour avoir un flux dual 2D
              DualphiFlux2(cCells) *=
                  (varlp->faceLength(flFaces) + varlp->faceLength(frFaces)) *
                  0.5;

              if (voisinage_pure)
                computeFluxPPPure(
                    gradPhi2(cCells), varlp->Phi(cCells), varlp->Phi(cfCells),
                    varlp->Phi(cbCells), HvLagrange(cCells),
                    HvLagrange(cfCells), HvLagrange(cbCells), Flux_sortant_av,
                    gt->deltat_n, 1, cCells, options->threshold,
                    limiteurs->projectionPlateauPenteComplet, flux_dual,
                    calcul_flux_dual, &deltaPhiFaceAv(cCells), &Bidon2(cCells));
              else
                computeFluxPP(
                    gradPhi2(cCells), varlp->Phi(cCells), varlp->Phi(cfCells),
                    varlp->Phi(cbCells), HvLagrange(cCells),
                    HvLagrange(cfCells), HvLagrange(cbCells), Flux_sortant_av,
                    gt->deltat_n, 1, cCells, options->threshold,
                    limiteurs->projectionPlateauPenteComplet, flux_dual,
                    calcul_flux_dual, &deltaPhiFaceAv(cCells), &Bidon2(cCells));
            }
          });
    }
  }
}
/**
 *******************************************************************************
 * \file computeUpwindFaceQuantitiesForProjection2()
 * \brief  phase 2 de projection : troisieme etape
 *        calcul de phiFace1
 *     qui contient la valeur reconstruite à l'ordre 1, 2 ou 3 des variables
 *projetees qui contient les flux des variables projetees avec l'option
 *pente-borne \param \return phiFace2
 *******************************************************************************
 */
void Remap::computeUpwindFaceQuantitiesForProjection2() noexcept {
  if (varlp->x_then_y_n) {
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
          if (options->projectionOrder <= 2) {
            if (limiteurs->projectionAvecPlateauPente == 0) {
              phiFace2(fFaces) = computeUpwindFaceQuantities(
                  varlp->faceNormal(fFaces), varlp->faceNormalVelocity(fFaces),
                  varlp->deltaxLagrange(fFaces), varlp->Xf(fFaces),
                  (Uremap1(cbCells) / varlp->vLagrange(cbCells)),
                  gradPhi2(cbCells), varlp->XcLagrange(cbCells),
                  (Uremap1(cfCells) / varlp->vLagrange(cfCells)),
                  gradPhi2(cfCells), varlp->XcLagrange(cfCells));
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
                (Uremap1(cfffCells) / varlp->vLagrange(cfffCells)),
                (Uremap1(cffCells) / varlp->vLagrange(cffCells)),
                (Uremap1(cfCells) / varlp->vLagrange(cfCells)),
                (Uremap1(cbCells) / varlp->vLagrange(cbCells)),
                (Uremap1(cbbCells) / varlp->vLagrange(cbbCells)),
                (Uremap1(cbbbCells) / varlp->vLagrange(cbbbCells)),
                HvLagrange(cfffCells), HvLagrange(cffCells),
                HvLagrange(cfCells), HvLagrange(cbCells), HvLagrange(cbbCells),
                HvLagrange(cbbbCells), varlp->faceNormalVelocity(fFaces),
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
          if (options->projectionOrder <= 2) {
            if (limiteurs->projectionAvecPlateauPente == 0) {
              phiFace2(fFaces) = computeUpwindFaceQuantities(
                  varlp->faceNormal(fFaces), varlp->faceNormalVelocity(fFaces),
                  varlp->deltaxLagrange(fFaces), varlp->Xf(fFaces),
                  (Uremap1(cbCells) / varlp->vLagrange(cbCells)),
                  gradPhi2(cbCells), varlp->XcLagrange(cbCells),
                  (Uremap1(cfCells) / varlp->vLagrange(cfCells)),
                  gradPhi2(cfCells), varlp->XcLagrange(cfCells));
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
                (Uremap1(cbbbCells) / varlp->vLagrange(cbbbCells)),
                (Uremap1(cbbCells) / varlp->vLagrange(cbbCells)),
                (Uremap1(cbCells) / varlp->vLagrange(cbCells)),
                (Uremap1(cfCells) / varlp->vLagrange(cfCells)),
                (Uremap1(cffCells) / varlp->vLagrange(cffCells)),
                (Uremap1(cfffCells) / varlp->vLagrange(cfffCells)),
                HvLagrange(cbbbCells), HvLagrange(cbbCells),
                HvLagrange(cbCells), HvLagrange(cfCells), HvLagrange(cffCells),
                HvLagrange(cfffCells), varlp->faceNormalVelocity(fFaces),
                gt->deltat_n);
          }
          // std::cout << " face " << fFaces << " " << cfCells << "  " <<
          // cbCells << " flux " << phiFace2(fFaces) << std::endl;
        });
  }
}
/**
 *******************************************************************************
 * \file computeUremap2()
 * \brief phase 2 de projection : etape finale
 *        calcul de la variable Uremap2 par ajout ou retrait des flux
          Mises à jour de l'indicateur mailles mixtes
          calcul de la valeur de Phi issu de Uremap1
 * \param
 * \return varlp->Uremap2, varlp->mixte, varlp->pure, varlp->Phi
 *******************************************************************************
 */
void Remap::computeUremap2() noexcept {
  RealArray1D<dim> exy = xThenYToDirection(!(varlp->x_then_y_n));
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

            if ((cstmesh->cylindrical_mesh != 1) ||
                dot(varlp->faceNormal(fFaces), exy) > 0.) {
              // stockage des flux aux faces pour la quantite de mouvement de
              // Vnr
              FluxFace2(cCells, fFacesOfCellC) = (computeRemapFlux(
                  options->projectionOrder,
                  limiteurs->projectionAvecPlateauPente,
                  varlp->faceNormalVelocity(fFaces), varlp->faceNormal(fFaces),
                  varlp->faceLength(fFaces), phiFace2(fFaces),
                  varlp->outerFaceNormal(cCells, fFacesOfCellC), exy,
                  gt->deltat_n));
              reduction9 = reduction9 +
                           (computeRemapFlux(
                               options->projectionOrder,
                               limiteurs->projectionAvecPlateauPente,
                               varlp->faceNormalVelocity(fFaces),
                               varlp->faceNormal(fFaces),
                               varlp->faceLength(fFaces), phiFace2(fFaces),
                               varlp->outerFaceNormal(cCells, fFacesOfCellC),
                               exy, gt->deltat_n));
              //
              //
            }
          }
          if (cdl->FluxBC > 0) {
            // flux exterieur

            reduction9 = reduction9 + (computeBoundaryFluxes(2, cCells, exy));
          }
        }

        varlp->Uremap2(cCells) = Uremap1(cCells) - reduction9;

        int nbmat = options->nbmat;
        for (int imat = 0; imat < nbmat; imat++) {
          if (varlp->Uremap2(cCells)[nbmat + imat] < 0.) {
            if (abs(varlp->Uremap2(cCells)[nbmat + imat]) >
                1.e2 * options->threshold)
              std::cout << " cell " << cCells
                        << " proj 2 --masse tres faiblement negative   "
                        << " avant " << varlp->Uremap2(cCells - 1)[nbmat + imat]
                        << " cell " << varlp->Uremap2(cCells)[nbmat + imat]
                        << " et volume " << varlp->Uremap2(cCells)[imat]
                        << " apres " << varlp->Uremap2(cCells + 1)[nbmat + imat]
                        << std::endl;
            varlp->Uremap2(cCells)[nbmat + imat] = 0.;
          }
          if (varlp->Uremap2(cCells)[2 * nbmat + imat] < 0.) {
            if (abs(varlp->Uremap2(cCells)[2 * nbmat + imat]) >
                1.e2 * options->threshold)
              std::cout << " cell " << cCells
                        << " proj 2 --energie tres faiblement negative "
                        << " avant "
                        << varlp->Uremap2(cCells - 1)[2 * nbmat + imat]
                        << " cell " << varlp->Uremap2(cCells)[2 * nbmat + imat]
                        << " apres "
                        << varlp->Uremap2(cCells + 1)[2 * nbmat + imat]
                        << std::endl;
            varlp->Uremap2(cCells)[2 * nbmat + imat] = 0.;
          }
        }
      });
}
