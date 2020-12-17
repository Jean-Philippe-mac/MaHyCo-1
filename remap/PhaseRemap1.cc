#include <math.h>  // for sqrt

#include <Kokkos_Core.hpp>
#include <algorithm>  // for copy
#include <array>      // for array
#include <iostream>   // for operator<<, basic_ostream::operat...
#include <vector>     // for allocator, vector

#include "../includes/VariablesLagRemap.h"
#include "Remap.h"                 // for Remap, Remap::Opt...
#include "UtilesRemap-Impl.h"      // for Remap::computeFluxPP
#include "mesh/CartesianMesh2D.h"  // for CartesianMesh2D
#include "types/MathFunctions.h"   // for dot
#include "types/MultiArray.h"      // for operator<<
#include "utils/Utils.h"           // for indexOf
/**
 *******************************************************************************
 * \file computeGradPhiFace1()
 * \brief phase 1 de projection : premiere etapes
 *  calcul des gradients aux faces verticales ou horizontales suivant le cas
 *  calcul des longueurs des faces verticales ou horizontales suivant le cas
 *  calcul des largeurs de cellule dans le sens verticales ou horizontales
 *suivant le cas \param \return gradPhiFace1, LfLagrange, HvLagrange
 *******************************************************************************
 */
void Remap::computeGradPhiFace1() noexcept {
  if (options->projectionOrder > 1) {
    if (varlp->x_then_y_n) {
      auto innerVerticalFaces(mesh->getInnerVerticalFaces());
      int nbInnerVerticalFaces(mesh->getNbInnerVerticalFaces());
      Kokkos::parallel_for(
          "computeGradPhiFace1", nbInnerVerticalFaces,
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
            gradPhiFace1(fFaces) = (varlp->Phi(cfCells) - varlp->Phi(cbCells)) /
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
          "computeGradPhiFace1", nbCells, KOKKOS_LAMBDA(const int& cCells) {
            int cId(cCells);
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
    } else {
      auto innerHorizontalFaces(mesh->getInnerHorizontalFaces());
      int nbInnerHorizontalFaces(mesh->getNbInnerHorizontalFaces());
      Kokkos::parallel_for(
          "computeGradPhiFace1", nbInnerHorizontalFaces,
          KOKKOS_LAMBDA(const int& fInnerHorizontalFaces) {
            size_t fId(innerHorizontalFaces[fInnerHorizontalFaces]);
            int fFaces(utils::indexOf(mesh->getFaces(), fId));
            int cfFrontCellF(mesh->getFrontCell(fId));
            int cfId(cfFrontCellF);
            int cfCells(cfId);
            int cbBackCellF(mesh->getBackCell(fId));
            int cbId(cbBackCellF);
            int cbCells(cbId);
            //
            gradPhiFace1(fFaces) = (varlp->Phi(cfCells) - varlp->Phi(cbCells)) /
                                   varlp->deltaxLagrange(fFaces);
            int n1FirstNodeOfFaceF(mesh->getFirstNodeOfFace(fId));
            int n1Id(n1FirstNodeOfFaceF);
            int n1Nodes(n1Id);
            int n2SecondNodeOfFaceF(mesh->getSecondNodeOfFace(fId));
            int n2Id(n2SecondNodeOfFaceF);
            int n2Nodes(n2Id);
            //
            LfLagrange(fFaces) = sqrt(
                (varlp->XLagrange(n1Nodes)[0] - varlp->XLagrange(n2Nodes)[0]) *
                    (varlp->XLagrange(n1Nodes)[0] -
                     varlp->XLagrange(n2Nodes)[0]) +
                (varlp->XLagrange(n1Nodes)[1] - varlp->XLagrange(n2Nodes)[1]) *
                    (varlp->XLagrange(n1Nodes)[1] -
                     varlp->XLagrange(n2Nodes)[1]));
          });
      Kokkos::parallel_for(
          "computeGradPhiFace1", nbCells, KOKKOS_LAMBDA(const int& cCells) {
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
    }
  }
}
/**
 *******************************************************************************
 * \file computeGradPhi()
 * \brief phase 1 de projection : seconde etape
 *        calcul du gradient aux mailles limites
 *        calcul des flux pente-borne arriere ou avant aux mailles
 *                 à partir du gradient precedent
 * \param
 * \return gradPhi1, deltaPhiFaceAr, deltaPhiFaceAv
 *******************************************************************************
 */
void Remap::computeGradPhi1() noexcept {
  // std::cout << " ordre de la projection " << options->projectionOrder <<
  // std::endl;
  if (options->projectionOrder > 1) {
    if (varlp->x_then_y_n) {
      // std::cout << " Phase 1 Horizontale computeGradPhi1 " << std::endl;
      Kokkos::parallel_for(
          "computeGradPhi1", nbCells, KOKKOS_LAMBDA(const int& cCells) {
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
            // std::cout << " Phase 1 Horizontale " << std::endl;
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

            gradPhi1(cCells) = computeAndLimitGradPhi(
                limiter, gradPhiFace1(frFaces), gradPhiFace1(flFaces),
                varlp->Phi(cCells), varlp->Phi(cfCells), varlp->Phi(cbCells),
                HvLagrange(cCells), HvLagrange(cfCells), HvLagrange(cbCells));

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
                    gradPhi1(cCells), varlp->Phi(cCells), varlp->Phi(cfCells),
                    varlp->Phi(cbCells), HvLagrange(cCells),
                    HvLagrange(cfCells), HvLagrange(cbCells), Flux_sortant_ar,
                    gt->deltat_n, 0, cCells, options->threshold,
                    limiteurs->projectionPlateauPenteComplet, flux_dual,
                    calcul_flux_dual, &deltaPhiFaceAr(cCells),
                    &DualphiFlux1(cCells));
              else
                computeFluxPP(gradPhi1(cCells), varlp->Phi(cCells),
                              varlp->Phi(cfCells), varlp->Phi(cbCells),
                              HvLagrange(cCells), HvLagrange(cfCells),
                              HvLagrange(cbCells), Flux_sortant_ar,
                              gt->deltat_n, 0, cCells, options->threshold,
                              limiteurs->projectionPlateauPenteComplet,
                              flux_dual, calcul_flux_dual,
                              &deltaPhiFaceAr(cCells), &DualphiFlux1(cCells));

              // pour avoir un flux dual 2D
              DualphiFlux1(cCells) *=
                  (varlp->faceLength(flFaces) + varlp->faceLength(frFaces)) *
                  0.5;

              if (voisinage_pure)
                computeFluxPPPure(
                    gradPhi1(cCells), varlp->Phi(cCells), varlp->Phi(cfCells),
                    varlp->Phi(cbCells), HvLagrange(cCells),
                    HvLagrange(cfCells), HvLagrange(cbCells), Flux_sortant_av,
                    gt->deltat_n, 1, cCells, options->threshold,
                    limiteurs->projectionPlateauPenteComplet, flux_dual,
                    calcul_flux_dual, &deltaPhiFaceAv(cCells), &Bidon1(cCells));
              else
                computeFluxPP(
                    gradPhi1(cCells), varlp->Phi(cCells), varlp->Phi(cfCells),
                    varlp->Phi(cbCells), HvLagrange(cCells),
                    HvLagrange(cfCells), HvLagrange(cbCells), Flux_sortant_av,
                    gt->deltat_n, 1, cCells, options->threshold,
                    limiteurs->projectionPlateauPenteComplet, flux_dual,
                    calcul_flux_dual, &deltaPhiFaceAv(cCells), &Bidon1(cCells));
            }
          });
    } else {
      // std::cout << " Phase 1 Verticale computeGradPhi1 " << std::endl;
      Kokkos::parallel_for(
          "computeGradPhi1", nbCells, KOKKOS_LAMBDA(const int& cCells) {
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

            gradPhi1(cCells) = computeAndLimitGradPhi(
                limiter, gradPhiFace1(fbFaces), gradPhiFace1(ftFaces),
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
                    gradPhi1(cCells), varlp->Phi(cCells), varlp->Phi(cbCells),
                    varlp->Phi(cfCells), HvLagrange(cCells),
                    HvLagrange(cbCells), HvLagrange(cfCells), Flux_sortant_av,
                    gt->deltat_n, 0, cCells, options->threshold,
                    limiteurs->projectionPlateauPenteComplet, flux_dual,
                    calcul_flux_dual, &deltaPhiFaceAr(cCells),
                    &DualphiFlux1(cCells));
              else
                computeFluxPP(gradPhi1(cCells), varlp->Phi(cCells),
                              varlp->Phi(cbCells), varlp->Phi(cfCells),
                              HvLagrange(cCells), HvLagrange(cbCells),
                              HvLagrange(cfCells), Flux_sortant_av,
                              gt->deltat_n, 0, cCells, options->threshold,
                              limiteurs->projectionPlateauPenteComplet,
                              flux_dual, calcul_flux_dual,
                              &deltaPhiFaceAr(cCells), &DualphiFlux1(cCells));

              // pour avoir un flux dual 2D
              DualphiFlux1(cCells) *=
                  (varlp->faceLength(fbFaces) + varlp->faceLength(ftFaces)) *
                  0.5;

              if (voisinage_pure)
                computeFluxPPPure(
                    gradPhi1(cCells), varlp->Phi(cCells), varlp->Phi(cbCells),
                    varlp->Phi(cfCells), HvLagrange(cCells),
                    HvLagrange(cbCells), HvLagrange(cfCells), Flux_sortant_ar,
                    gt->deltat_n, 1, cCells, options->threshold,
                    limiteurs->projectionPlateauPenteComplet, flux_dual,
                    calcul_flux_dual, &deltaPhiFaceAv(cCells), &Bidon1(cCells));
              else
                computeFluxPP(
                    gradPhi1(cCells), varlp->Phi(cCells), varlp->Phi(cbCells),
                    varlp->Phi(cfCells), HvLagrange(cCells),
                    HvLagrange(cbCells), HvLagrange(cfCells), Flux_sortant_ar,
                    gt->deltat_n, 1, cCells, options->threshold,
                    limiteurs->projectionPlateauPenteComplet, flux_dual,
                    calcul_flux_dual, &deltaPhiFaceAv(cCells), &Bidon1(cCells));
            }
          });
    }
  }
}
/**
 *******************************************************************************
 * \file computeUpwindFaceQuantitiesForProjection1()
 * \brief  phase 1 de projection : troisieme etape
 *        calcul de phiFace1
 *     qui contient la valeur reconstruite à l'ordre 1, 2 ou 3 des variables
 *projetees qui contient les flux des variables projetees avec l'option
 *pente-borne \param \return phiFace1
 *******************************************************************************
 */
void Remap::computeUpwindFaceQuantitiesForProjection1() noexcept {
  if (varlp->x_then_y_n) {
    // std::cout << " Phase Projection 1 Horizontale " << std::endl;
    auto innerVerticalFaces(mesh->getInnerVerticalFaces());
    int nbInnerVerticalFaces(mesh->getNbInnerVerticalFaces());
    Kokkos::parallel_for(
        "computeUpwindFaceQuantitiesForProjection1", nbInnerVerticalFaces,
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
          // à la valeur du flux (integration de phi(x)) pour l'ordre 2 avec
          // Plateau-Pente à la valeur du flux (integration de phi(x)) pour
          // l'ordre 3
          if (options->projectionOrder <= 2) {
            if (limiteurs->projectionAvecPlateauPente == 0) {
              phiFace1(fFaces) = computeUpwindFaceQuantities(
                  varlp->faceNormal(fFaces), varlp->faceNormalVelocity(fFaces),
                  varlp->deltaxLagrange(fFaces), varlp->Xf(fFaces),
                  (varlp->ULagrange(cbCells) / varlp->vLagrange(cbCells)),
                  gradPhi1(cbCells), varlp->XcLagrange(cbCells),
                  (varlp->ULagrange(cfCells) / varlp->vLagrange(cfCells)),
                  gradPhi1(cfCells), varlp->XcLagrange(cfCells));
            } else {
              phiFace1(fFaces) =
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

            phiFace1(fFaces) = computeVecFluxOrdre3(
                (varlp->ULagrange(cbbbCells) / varlp->vLagrange(cbbbCells)),
                (varlp->ULagrange(cbbCells) / varlp->vLagrange(cbbCells)),
                (varlp->ULagrange(cbCells) / varlp->vLagrange(cbCells)),
                (varlp->ULagrange(cfCells) / varlp->vLagrange(cfCells)),
                (varlp->ULagrange(cffCells) / varlp->vLagrange(cffCells)),
                (varlp->ULagrange(cfffCells) / varlp->vLagrange(cfffCells)),
                HvLagrange(cbbbCells), HvLagrange(cbbCells),
                HvLagrange(cbCells), HvLagrange(cfCells), HvLagrange(cffCells),
                HvLagrange(cfffCells), varlp->faceNormalVelocity(fFaces),
                gt->deltat_n);
          }
        });
  } else {
    // std::cout << " Phase Projection 1 Verticale " << std::endl;
    auto innerHorizontalFaces(mesh->getInnerHorizontalFaces());
    int nbInnerHorizontalFaces(mesh->getNbInnerHorizontalFaces());
    Kokkos::parallel_for(
        "computeUpwindFaceQuantitiesForProjection1", nbInnerHorizontalFaces,
        KOKKOS_LAMBDA(const int& fInnerHorizontalFaces) {
          size_t fId(innerHorizontalFaces[fInnerHorizontalFaces]);
          int fFaces(utils::indexOf(mesh->getFaces(), fId));
          int cfFrontCellF(mesh->getFrontCell(fId));
          int cfId(cfFrontCellF);
          int cfCells(cfId);
          int cbBackCellF(mesh->getBackCell(fId));
          int cbId(cbBackCellF);
          int cbCells(cbId);
          // phiFace1 correspond
          // à la valeur de phi(x) à la face pour l'ordre 2 sans plateau pente
          // à la valeur du flux (integration de phi(x)) pour l'ordre 2 avec
          // Plateau-Pente à la valeur du flux (integration de phi(x)) pour
          // l'ordre 3
          if (options->projectionOrder <= 2) {
            if (limiteurs->projectionAvecPlateauPente == 0) {
              phiFace1(fFaces) = computeUpwindFaceQuantities(
                  varlp->faceNormal(fFaces), varlp->faceNormalVelocity(fFaces),
                  varlp->deltaxLagrange(fFaces), varlp->Xf(fFaces),
                  (varlp->ULagrange(cbCells) / varlp->vLagrange(cbCells)),
                  gradPhi1(cbCells), varlp->XcLagrange(cbCells),
                  (varlp->ULagrange(cfCells) / varlp->vLagrange(cfCells)),
                  gradPhi1(cfCells), varlp->XcLagrange(cfCells));
            } else {
              phiFace1(fFaces) =
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

            phiFace1(fFaces) = computeVecFluxOrdre3(
                (varlp->ULagrange(cfffCells) / varlp->vLagrange(cfffCells)),
                (varlp->ULagrange(cffCells) / varlp->vLagrange(cffCells)),
                (varlp->ULagrange(cfCells) / varlp->vLagrange(cfCells)),
                (varlp->ULagrange(cbCells) / varlp->vLagrange(cbCells)),
                (varlp->ULagrange(cbbCells) / varlp->vLagrange(cbbCells)),
                (varlp->ULagrange(cbbbCells) / varlp->vLagrange(cbbbCells)),
                HvLagrange(cfffCells), HvLagrange(cffCells),
                HvLagrange(cfCells), HvLagrange(cbCells), HvLagrange(cbbCells),
                HvLagrange(cbbbCells), varlp->faceNormalVelocity(fFaces),
                gt->deltat_n);
          }
        });
  }
}
/**
 *******************************************************************************
 * \file computeUremap1()
 * \brief phase 1 de projection : etape finale
 *        calcul de la variable Uremap1 par ajout ou retrait des flux
          Mises à jour de l'indicateur mailles mixtes
          calcul de la valeur de Phi issu de Uremap1
 * \param
 * \return Uremap1, varlp->mixte, varlp->pure, varlp->Phi
 *******************************************************************************
 */
void Remap::computeUremap1() noexcept {
  int nbmat = options->nbmat;
  RealArray1D<dim> exy = xThenYToDirection(varlp->x_then_y_n);
  Kokkos::parallel_for(
      "computeUremap1", nbCells, KOKKOS_LAMBDA(const int& cCells) {
        int cId(cCells);

        // std::cout << " cCells " << cCells << std::endl;
        // std::cout << "ULagrange " << ULagrange(cCells) << std::endl;
        RealArray1D<nbequamax> reduction8 = Uzero;
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
              FluxFace1(cCells, fFacesOfCellC) = computeRemapFlux(
                  options->projectionOrder,
                  limiteurs->projectionAvecPlateauPente,
                  varlp->faceNormalVelocity(fFaces), varlp->faceNormal(fFaces),
                  varlp->faceLength(fFaces), phiFace1(fFaces),
                  varlp->outerFaceNormal(cCells, fFacesOfCellC), exy,
                  gt->deltat_n);
              reduction8 = reduction8 +
                           (computeRemapFlux(
                               options->projectionOrder,
                               limiteurs->projectionAvecPlateauPente,
                               varlp->faceNormalVelocity(fFaces),
                               varlp->faceNormal(fFaces),
                               varlp->faceLength(fFaces), phiFace1(fFaces),
                               varlp->outerFaceNormal(cCells, fFacesOfCellC),
                               exy, gt->deltat_n));
            }
          }
          if (cdl->FluxBC > 0) {
            // flux exterieur eventuel
            //
            reduction8 = reduction8 + (computeBoundaryFluxes(1, cCells, exy));
            //
          }
        }

        Uremap1(cCells) = varlp->ULagrange(cCells) - reduction8;

        for (int imat = 0; imat < nbmat; imat++) {
          if (Uremap1(cCells)[nbmat + imat] < 0.) {
            if (abs(Uremap1(cCells)[nbmat + imat]) > 1.e2 * options->threshold)
              std::cout << " cell " << cCells
                        << " proj 1 --masse tres faiblement negative   "
                        << " avant " << Uremap1(cCells - 1)[nbmat + imat]
                        << " cell " << Uremap1(cCells)[nbmat + imat]
                        << " et volume " << Uremap1(cCells)[imat] << " apres "
                        << Uremap1(cCells + 1)[nbmat + imat] << std::endl;
            Uremap1(cCells)[nbmat + imat] = 0.;
          }
          if (Uremap1(cCells)[2 * nbmat + imat] < 0.) {
            if (abs(Uremap1(cCells)[nbmat + imat]) > 1.e2 * options->threshold)
              std::cout << " cell " << cCells
                        << " --energie tres faiblement negative "
                        << " avant " << Uremap1(cCells - 1)[2 * nbmat + imat]
                        << " cell " << Uremap1(cCells)[2 * nbmat + imat]
                        << " apres " << Uremap1(cCells + 1)[2 * nbmat + imat]
                        << std::endl;
            Uremap1(cCells)[2 * nbmat + imat] = 0.;
          }
        }

        if (limiteurs->projectionAvecPlateauPente == 1) {
          // option ou on ne regarde pas la variation de rho, V et e
          // phi = (f1, f2, rho1*f1, rho2*f2, Vx, Vy, e1, e2
          // ce qui permet d'ecrire le flux telque
          // Flux = (dv1 = f1dv, dv2=f2*dv, dm1=rho1*df1, dm2=rho2*df2, d(mVx) =
          // Vx*(dm1+dm2), d(mVy) = Vy*(dm1+dm2), d(m1e1) = e1*dm1,  d(m2e2) =
          // e2*dm2 dans computeFluxPP

          double somme_volume = 0.;
          for (int imat = 0; imat < nbmat; imat++) {
            somme_volume += Uremap1(cCells)[imat];
          }
          // Phi volume
          double somme_masse = 0.;
          for (int imat = 0; imat < nbmat; imat++) {
            varlp->Phi(cCells)[imat] = Uremap1(cCells)[imat] / somme_volume;
            // Phi masse
            if (Uremap1(cCells)[imat] != 0.)
              varlp->Phi(cCells)[nbmat + imat] =
                  Uremap1(cCells)[nbmat + imat] / (Uremap1(cCells)[imat]);
            else
              varlp->Phi(cCells)[nbmat + imat] = 0.;
            somme_masse += Uremap1(cCells)[nbmat + imat];
          }
          // Phi Vitesse
          varlp->Phi(cCells)[3 * nbmat] =
              Uremap1(cCells)[3 * nbmat] / somme_masse;
          varlp->Phi(cCells)[3 * nbmat + 1] =
              Uremap1(cCells)[3 * nbmat + 1] / somme_masse;
          // Phi energie
          for (int imat = 0; imat < nbmat; imat++) {
            if (Uremap1(cCells)[nbmat + imat] != 0.)
              varlp->Phi(cCells)[2 * nbmat + imat] =
                  Uremap1(cCells)[2 * nbmat + imat] /
                  Uremap1(cCells)[nbmat + imat];
            else
              varlp->Phi(cCells)[2 * nbmat + imat] = 0.;
          }
          // Phi energie cinétique
          if (options->projectionConservative == 1)
            varlp->Phi(cCells)[3 * nbmat + 2] =
                Uremap1(cCells)[3 * nbmat + 2] / somme_masse;

        } else {
          varlp->Phi(cCells) = Uremap1(cCells) / varlp->vLagrange(cCells);
        }

        // Mises à jour de l'indicateur mailles mixtes
        int matcell(0);
        int imatpure(-1);
        for (int imat = 0; imat < nbmat; imat++)
          if (varlp->Phi(cCells)[imat] > 0.) {
            matcell++;
            imatpure = imat;
          }
        if (matcell > 1) {
          varlp->mixte(cCells) = 1;
          varlp->pure(cCells) = -1;
        } else {
          varlp->mixte(cCells) = 0;
          varlp->pure(cCells) = imatpure;
        }
      });
}
