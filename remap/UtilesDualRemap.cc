#include <math.h>    // for fabs
#include <stdlib.h>  // for abs

#include <array>  // for array

#include "Remap.h"                // for Remap, Remap::Options
#include "types/MathFunctions.h"  // for min, max
#include "utils/Utils.h"          // for indexOf
/**
 *******************************************************************************
 * \file getRightAndLeftFluxMasse1
 * \brief calcul des flux de masse sur les faces virtuelles
 *       gauche et droite de la maille duale
 *       pour l'etape 1 de la projection
 * \param  FluxFace1 (flux de masse aux mailles)
 * \return RightFluxMasse, LeftFluxMasse
 *******************************************************************************
 */
void Remap::getRightAndLeftFluxMasse1(const int nbmat, const size_t pNodes) {
  // construction des mailles et faces associées pour recuperer les
  // flux de masses aux faces à gauche et à droite
#include "FacesLeftAndRight.h"
  // on prend moyenne les flux de masses (nbmat + imat)
  // des 4 faces verticales des 2 mailles à droite du noeud
  // ou
  // des 4 faces verticales des 2 mailles à gauche du noeud
  RightFluxMasse(pNodes) = 0.;
  LeftFluxMasse(pNodes) = 0.;
  for (int imat = 0; imat < nbmat; imat++) {
    RightFluxMassePartielle(pNodes)[imat] = 0;
    LeftFluxMassePartielle(pNodes)[imat] = 0;

    if (VerticalFaceOfNode(pNodes)[0] != -1) {
      RightFluxMassePartielle(pNodes)[imat] +=
          (FluxFace1(cfCell1, fOfcfCell1)[nbmat + imat] *
               varlp->outerFaceNormal(cfCell1, fOfcfCell1)[0] +
           FluxFace1(cfCell1, frOfcfCell1)[nbmat + imat] *
               varlp->outerFaceNormal(cfCell1, frOfcfCell1)[0]);
      LeftFluxMassePartielle(pNodes)[imat] +=
          (FluxFace1(cbCell1, fOfcbCell1)[nbmat + imat] *
               varlp->outerFaceNormal(cbCell1, fOfcbCell1)[0] +
           FluxFace1(cbCell1, flOfcbCell1)[nbmat + imat] *
               varlp->outerFaceNormal(cbCell1, flOfcbCell1)[0]);
    }

    if (VerticalFaceOfNode(pNodes)[1] != -1) {
      RightFluxMassePartielle(pNodes)[imat] +=
          (FluxFace1(cfCell2, fOfcfCell2)[nbmat + imat] *
               varlp->outerFaceNormal(cfCell2, fOfcfCell2)[0] +
           FluxFace1(cfCell2, frOfcfCell2)[nbmat + imat] *
               varlp->outerFaceNormal(cfCell2, frOfcfCell2)[0]);
      LeftFluxMassePartielle(pNodes)[imat] +=
          (FluxFace1(cbCell2, fOfcbCell2)[nbmat + imat] *
               varlp->outerFaceNormal(cbCell2, fOfcbCell2)[0] +
           FluxFace1(cbCell2, flOfcbCell2)[nbmat + imat] *
               varlp->outerFaceNormal(cbCell2, flOfcbCell2)[0]);
    }

    if (nbfaces != 0) {
      RightFluxMassePartielle(pNodes)[imat] /= nbfaces;
      LeftFluxMassePartielle(pNodes)[imat] /= nbfaces;
    }

    RightFluxMasse(pNodes) += RightFluxMassePartielle(pNodes)[imat];
    LeftFluxMasse(pNodes) += LeftFluxMassePartielle(pNodes)[imat];
  }
}
/**
 *******************************************************************************
 * \file getRightAndLeftFluxMasse2
 * \brief calcul des flux de masse sur les faces virtuelles
 *       gauche et droite de la maille duale
 *       pour l'etape 2 de la projection
 * \param  FluxFace2 (flux de masse aux mailles)
 * \return RightFluxMasse, LeftFluxMasse
 *******************************************************************************
 */
void Remap::getRightAndLeftFluxMasse2(const int nbmat, const size_t pNodes) {
  // construction des mailles et faces associées pour recuperer les
  // flux de masses aux faces à gauche et à droite
#include "FacesLeftAndRight.h"
  // on prend moyenne les flux de masses (nbmat + imat)
  // des 4 faces verticales des 2 mailles à droite du noeud
  // ou
  // des 4 faces verticales des 2 mailles à gauche du noeud
  RightFluxMasse(pNodes) = 0.;
  LeftFluxMasse(pNodes) = 0.;
  for (int imat = 0; imat < nbmat; imat++) {
    RightFluxMassePartielle(pNodes)[imat] = 0;
    LeftFluxMassePartielle(pNodes)[imat] = 0;

    if (VerticalFaceOfNode(pNodes)[0] != -1) {
      RightFluxMassePartielle(pNodes)[imat] +=
          (FluxFace2(cfCell1, fOfcfCell1)[nbmat + imat] *
               varlp->outerFaceNormal(cfCell1, fOfcfCell1)[0] +
           FluxFace2(cfCell1, frOfcfCell1)[nbmat + imat] *
               varlp->outerFaceNormal(cfCell1, frOfcfCell1)[0]);
      LeftFluxMassePartielle(pNodes)[imat] +=
          (FluxFace2(cbCell1, fOfcbCell1)[nbmat + imat] *
               varlp->outerFaceNormal(cbCell1, fOfcbCell1)[0] +
           FluxFace2(cbCell1, flOfcbCell1)[nbmat + imat] *
               varlp->outerFaceNormal(cbCell1, flOfcbCell1)[0]);
    }

    if (VerticalFaceOfNode(pNodes)[1] != -1) {
      RightFluxMassePartielle(pNodes)[imat] +=
          (FluxFace2(cfCell2, fOfcfCell2)[nbmat + imat] *
               varlp->outerFaceNormal(cfCell2, fOfcfCell2)[0] +
           FluxFace2(cfCell2, frOfcfCell2)[nbmat + imat] *
               varlp->outerFaceNormal(cfCell2, frOfcfCell2)[0]);
      LeftFluxMassePartielle(pNodes)[imat] +=
          (FluxFace2(cbCell2, fOfcbCell2)[nbmat + imat] *
               varlp->outerFaceNormal(cbCell2, fOfcbCell2)[0] +
           FluxFace2(cbCell2, flOfcbCell2)[nbmat + imat] *
               varlp->outerFaceNormal(cbCell2, flOfcbCell2)[0]);
    }
    if (nbfaces != 0) {
      RightFluxMassePartielle(pNodes)[imat] /= nbfaces;
      LeftFluxMassePartielle(pNodes)[imat] /= nbfaces;
    }

    RightFluxMasse(pNodes) += RightFluxMassePartielle(pNodes)[imat];
    LeftFluxMasse(pNodes) += LeftFluxMassePartielle(pNodes)[imat];
  }
}
/**
 *******************************************************************************
 * \file getTopAndBottomFluxMasse1
 * \brief calcul des flux de masse sur les faces virtuelles
 *       haute et basse de la maille duale
 *       pour l'etape 1 de la projection
 * \param  FluxFace1 (flux de masse aux mailles)
 * \return TopFluxMasse, BottomFluxMasse
 *******************************************************************************
 */
void Remap::getTopAndBottomFluxMasse1(const int nbmat, const size_t pNodes) {
  // construction des mailles et faces associées pour recuperer les
  // flux de masses aux faces dessus et dessous
#include "FacesTopAndBottom.h"
  // on prend moyenne les flux de masses (nbmat + imat)
  // des 4 faces verticales des 2 mailles à droite du noeud
  // ou
  // des 4 faces verticales des 2 mailles à gauche du noeud

  TopFluxMasse(pNodes) = 0.;
  BottomFluxMasse(pNodes) = 0;
  for (int imat = 0; imat < nbmat; imat++) {
    TopFluxMassePartielle(pNodes)[imat] = 0.;
    BottomFluxMassePartielle(pNodes)[imat] = 0;

    if (HorizontalFaceOfNode(pNodes)[0] != -1) {
      TopFluxMassePartielle(pNodes)[imat] +=
          (FluxFace1(cbCell1, fOfcbCell1)[nbmat + imat] *
               varlp->outerFaceNormal(cbCell1, fOfcbCell1)[1] +
           FluxFace1(cbCell1, ftOfcbCell1)[nbmat + imat] *
               varlp->outerFaceNormal(cbCell1, ftOfcbCell1)[1]);
      BottomFluxMassePartielle(pNodes)[imat] +=
          (FluxFace1(cfCell1, fOfcfCell1)[nbmat + imat] *
               varlp->outerFaceNormal(cfCell1, fOfcfCell1)[1] +
           FluxFace1(cfCell1, fbOfcfCell1)[nbmat + imat] *
               varlp->outerFaceNormal(cfCell1, fbOfcfCell1)[1]);
    }
    if (HorizontalFaceOfNode(pNodes)[1] != -1) {
      TopFluxMassePartielle(pNodes)[imat] +=
          (FluxFace1(cbCell2, fOfcbCell2)[nbmat + imat] *
               varlp->outerFaceNormal(cbCell2, fOfcbCell2)[1] +
           FluxFace1(cbCell2, ftOfcbCell2)[nbmat + imat] *
               varlp->outerFaceNormal(cbCell2, ftOfcbCell2)[1]);
      BottomFluxMassePartielle(pNodes)[imat] +=
          (FluxFace1(cfCell2, fOfcfCell2)[nbmat + imat] *
               varlp->outerFaceNormal(cfCell2, fOfcfCell2)[1] +
           FluxFace1(cfCell2, fbOfcfCell2)[nbmat + imat] *
               varlp->outerFaceNormal(cfCell2, fbOfcfCell2)[1]);
    }
    if (nbfaces != 0) {
      TopFluxMassePartielle(pNodes)[imat] /= nbfaces;
      BottomFluxMassePartielle(pNodes)[imat] /= nbfaces;
    }

    TopFluxMasse(pNodes) += TopFluxMassePartielle(pNodes)[imat];
    BottomFluxMasse(pNodes) += BottomFluxMassePartielle(pNodes)[imat];
  }
}
/**
 *******************************************************************************
 * \file getTopAndBottomFluxMasse2
 * \brief calcul des flux de masse sur les faces virtuelles
 *       haute et basse de la maille duale
 *       pour l'etape 2 de la projection
 * \param  FluxFace2 (flux de masse aux mailles)
 * \return TopFluxMasse, BottomFluxMasse
 *******************************************************************************
 */
void Remap::getTopAndBottomFluxMasse2(const int nbmat, const size_t pNodes) {
  // construction des mailles et faces associées pour recuperer les
  // flux de masses aux faces dessus et dessous
#include "FacesTopAndBottom.h"
  // on prend moyenne les flux de masses (nbmat + imat)
  // des 4 faces verticales des 2 mailles à droite du noeud
  // ou
  // des 4 faces verticales des 2 mailles à gauche du noeud
  TopFluxMasse(pNodes) = 0.;
  BottomFluxMasse(pNodes) = 0;
  for (int imat = 0; imat < nbmat; imat++) {
    TopFluxMassePartielle(pNodes)[imat] = 0.;
    BottomFluxMassePartielle(pNodes)[imat] = 0;

    if (HorizontalFaceOfNode(pNodes)[0] != -1) {
      TopFluxMassePartielle(pNodes)[imat] +=
          (FluxFace2(cbCell1, fOfcbCell1)[nbmat + imat] *
               varlp->outerFaceNormal(cbCell1, fOfcbCell1)[1] +
           FluxFace2(cbCell1, ftOfcbCell1)[nbmat + imat] *
               varlp->outerFaceNormal(cbCell1, ftOfcbCell1)[1]);
      BottomFluxMassePartielle(pNodes)[imat] +=
          (FluxFace2(cfCell1, fOfcfCell1)[nbmat + imat] *
               varlp->outerFaceNormal(cfCell1, fOfcfCell1)[1] +
           FluxFace2(cfCell1, fbOfcfCell1)[nbmat + imat] *
               varlp->outerFaceNormal(cfCell1, fbOfcfCell1)[1]);
    }
    if (HorizontalFaceOfNode(pNodes)[1] != -1) {
      TopFluxMassePartielle(pNodes)[imat] +=
          (FluxFace2(cbCell2, fOfcbCell2)[nbmat + imat] *
               varlp->outerFaceNormal(cbCell2, fOfcbCell2)[1] +
           FluxFace2(cbCell2, ftOfcbCell2)[nbmat + imat] *
               varlp->outerFaceNormal(cbCell2, ftOfcbCell2)[1]);
      BottomFluxMassePartielle(pNodes)[imat] +=
          (FluxFace2(cfCell2, fOfcfCell2)[nbmat + imat] *
               varlp->outerFaceNormal(cfCell2, fOfcfCell2)[1] +
           FluxFace2(cfCell2, fbOfcfCell2)[nbmat + imat] *
               varlp->outerFaceNormal(cfCell2, fbOfcfCell2)[1]);
    }
    if (nbfaces != 0) {
      TopFluxMassePartielle(pNodes)[imat] /= nbfaces;
      BottomFluxMassePartielle(pNodes)[imat] /= nbfaces;
    }

    TopFluxMasse(pNodes) += TopFluxMassePartielle(pNodes)[imat];
    BottomFluxMasse(pNodes) += BottomFluxMassePartielle(pNodes)[imat];
  }
}
void Remap::getBottomUpwindVelocity(const size_t BottomNode, const size_t pNode,
                                    RealArray1D<nbequamax> gradDualPhimoins,
                                    RealArray1D<nbequamax> gradDualPhi0) {
  int order2 = options->projectionOrder - 1;
  int signbottom;
  double ubottom =
      0.5 * (varlp->DualPhi(BottomNode)[1] + varlp->DualPhi(pNode)[1]);

  if (ubottom > 0)
    signbottom = 1;
  else
    signbottom = -1;

  if (BottomFluxMasse(pNode) > 0) {
    BottomupwindVelocity(pNode)[0] =
        varlp->DualPhi(BottomNode)[0] +
        order2 * (0.5 * gradDualPhimoins[0] *
                  (signbottom * (varlp->XLagrange(pNode)[1] -
                                 varlp->XLagrange(BottomNode)[1]) -
                   gt->deltat_n * ubottom));
    BottomupwindVelocity(pNode)[1] =
        varlp->DualPhi(BottomNode)[1] +
        order2 * (0.5 * gradDualPhimoins[1] *
                  (signbottom * (varlp->XLagrange(pNode)[1] -
                                 varlp->XLagrange(BottomNode)[1]) -
                   gt->deltat_n * ubottom));
    // energie cinetique
    BottomupwindVelocity(pNode)[2] =
        varlp->DualPhi(BottomNode)[2] +
        order2 * (0.5 * gradDualPhimoins[2] *
                  (signbottom * (varlp->XLagrange(pNode)[1] -
                                 varlp->XLagrange(BottomNode)[1]) -
                   gt->deltat_n * ubottom));
  } else {
    BottomupwindVelocity(pNode)[0] =
        varlp->DualPhi(pNode)[0] +
        order2 * (0.5 * gradDualPhi0[0] *
                  (signbottom * (varlp->XLagrange(pNode)[1] -
                                 varlp->XLagrange(BottomNode)[1]) -
                   gt->deltat_n * ubottom));
    BottomupwindVelocity(pNode)[1] =
        varlp->DualPhi(pNode)[1] +
        order2 * (0.5 * gradDualPhi0[1] *
                  (signbottom * (varlp->XLagrange(pNode)[1] -
                                 varlp->XLagrange(BottomNode)[1]) -
                   gt->deltat_n * ubottom));
    // energie cinetique
    BottomupwindVelocity(pNode)[2] =
        varlp->DualPhi(pNode)[2] +
        order2 * (0.5 * gradDualPhi0[2] *
                  (signbottom * (varlp->XLagrange(pNode)[1] -
                                 varlp->XLagrange(BottomNode)[1]) -
                   gt->deltat_n * ubottom));
  }
}
void Remap::getTopUpwindVelocity(const size_t TopNode, const size_t pNode,
                                 RealArray1D<nbequamax> gradDualPhiplus,
                                 RealArray1D<nbequamax> gradDualPhi0) {
  int order2 = options->projectionOrder - 1;
  int signtop;
  double utop = 0.5 * (varlp->DualPhi(pNode)[1] + varlp->DualPhi(TopNode)[1]);

  if (utop > 0)
    signtop = 1;
  else
    signtop = -1;

  if (TopFluxMasse(pNode) < 0) {
    TopupwindVelocity(pNode)[0] =
        varlp->DualPhi(TopNode)[0] +
        order2 * (0.5 * gradDualPhiplus[0] *
                  (signtop * (varlp->XLagrange(TopNode)[1] -
                              varlp->XLagrange(pNode)[1]) -
                   gt->deltat_n * utop));
    TopupwindVelocity(pNode)[1] =
        varlp->DualPhi(TopNode)[1] +
        order2 * (0.5 * gradDualPhiplus[1] *
                  (signtop * (varlp->XLagrange(TopNode)[1] -
                              varlp->XLagrange(pNode)[1]) -
                   gt->deltat_n * utop));
    // energie cinetique
    TopupwindVelocity(pNode)[2] =
      varlp->DualPhi(TopNode)[2] +
      order2 * (0.5 * gradDualPhiplus[2] *
		(signtop * (varlp->XLagrange(TopNode)[1] -
			    varlp->XLagrange(pNode)[1]) -
		 gt->deltat_n * utop));
  } else {
    TopupwindVelocity(pNode)[0] =
        varlp->DualPhi(pNode)[0] +
        order2 * (0.5 * gradDualPhi0[0] *
                  (signtop * (varlp->XLagrange(TopNode)[1] -
                              varlp->XLagrange(pNode)[1]) -
                   gt->deltat_n * utop));
    TopupwindVelocity(pNode)[1] =
        varlp->DualPhi(pNode)[1] +
        order2 * (0.5 * gradDualPhi0[1] *
                  (signtop * (varlp->XLagrange(TopNode)[1] -
                              varlp->XLagrange(pNode)[1]) -
                   gt->deltat_n * utop));
    // energie cinetique
    TopupwindVelocity(pNode)[2] =
        varlp->DualPhi(pNode)[2] +
        order2 * (0.5 * gradDualPhi0[2] *
                  (signtop * (varlp->XLagrange(TopNode)[1] -
                              varlp->XLagrange(pNode)[1]) -
                   gt->deltat_n * utop));
    
  }
}

void Remap::getRightUpwindVelocity(const size_t RightNode, const size_t pNode,
                                   RealArray1D<nbequamax> gradDualPhiplus,
                                   RealArray1D<nbequamax> gradDualPhi0) {
  int order2 = options->projectionOrder - 1;
  int signright;
  double uright =
      0.5 * (varlp->DualPhi(pNode)[0] + varlp->DualPhi(RightNode)[0]);

  if (uright > 0)
    signright = 1;
  else
    signright = -1;

  if (RightFluxMasse(pNode) < 0) {
    RightupwindVelocity(pNode)[0] =
        varlp->DualPhi(RightNode)[0] +
        order2 * (0.5 * gradDualPhiplus[0] *
                  (signright * (varlp->XLagrange(RightNode)[0] -
                                varlp->XLagrange(pNode)[0]) -
                   gt->deltat_n * uright));

    RightupwindVelocity(pNode)[1] =
        varlp->DualPhi(RightNode)[1] +
        order2 * (0.5 * gradDualPhiplus[1] *
                  (signright * (varlp->XLagrange(RightNode)[0] -
                                varlp->XLagrange(pNode)[0]) -
                   gt->deltat_n * uright));
    // energie cinetique
    RightupwindVelocity(pNode)[2] =
        varlp->DualPhi(RightNode)[2] +
        order2 * (0.5 * gradDualPhiplus[2] *
                  (signright * (varlp->XLagrange(RightNode)[0] -
                                varlp->XLagrange(pNode)[0]) -
                   gt->deltat_n * uright));
    
  } else {
    RightupwindVelocity(pNode)[0] =
        varlp->DualPhi(pNode)[0] +
        order2 * (0.5 * gradDualPhi0[0] *
                  (signright * (varlp->XLagrange(RightNode)[0] -
                                varlp->XLagrange(pNode)[0]) -
                   gt->deltat_n * uright));
    RightupwindVelocity(pNode)[1] =
        varlp->DualPhi(pNode)[1] +
        order2 * (0.5 * gradDualPhi0[1] *
                  (signright * (varlp->XLagrange(RightNode)[0] -
                                varlp->XLagrange(pNode)[0]) -
                   gt->deltat_n * uright));
    // energie cinetique
    RightupwindVelocity(pNode)[2] =
      varlp->DualPhi(pNode)[2] +
      order2 * (0.5 * gradDualPhi0[2] *
		(signright * (varlp->XLagrange(RightNode)[0] -
			      varlp->XLagrange(pNode)[0]) -
		 gt->deltat_n * uright));
  }
}
void Remap::getLeftUpwindVelocity(const size_t LeftNode, const size_t pNode,
                                  RealArray1D<nbequamax> gradDualPhimoins,
                                  RealArray1D<nbequamax> gradDualPhi0) {
  int order2 = options->projectionOrder - 1;
  int signleft;
  double uleft = 0.5 * (varlp->DualPhi(LeftNode)[0] + varlp->DualPhi(pNode)[0]);

  if (uleft > 0)
    signleft = 1;
  else
    signleft = -1;

  if (LeftFluxMasse(pNode) > 0) {
    LeftupwindVelocity(pNode)[0] =
        varlp->DualPhi(LeftNode)[0] +
        order2 * (0.5 * gradDualPhimoins[0] *
                  (signleft * (varlp->XLagrange(pNode)[0] -
                               varlp->XLagrange(LeftNode)[0]) -
                   gt->deltat_n * uleft));
    LeftupwindVelocity(pNode)[1] =
        varlp->DualPhi(LeftNode)[1] +
        order2 * (0.5 * gradDualPhimoins[1] *
                  (signleft * (varlp->XLagrange(pNode)[0] -
                               varlp->XLagrange(LeftNode)[0]) -
                   gt->deltat_n * uleft));
    // energie cinetique
    LeftupwindVelocity(pNode)[2] =
        varlp->DualPhi(LeftNode)[2] +
        order2 * (0.5 * gradDualPhimoins[2] *
                  (signleft * (varlp->XLagrange(pNode)[0] -
                               varlp->XLagrange(LeftNode)[0]) -
                   gt->deltat_n * uleft));    
  } else {
    LeftupwindVelocity(pNode)[0] =
        varlp->DualPhi(pNode)[0] +
        order2 * (0.5 * gradDualPhi0[0] *
                  (signleft * (varlp->XLagrange(pNode)[0] -
                               varlp->XLagrange(LeftNode)[0]) -
                   gt->deltat_n * uleft));
    LeftupwindVelocity(pNode)[1] =
        varlp->DualPhi(pNode)[1] +
        order2 * (0.5 * gradDualPhi0[1] *
                  (signleft * (varlp->XLagrange(pNode)[0] -
                               varlp->XLagrange(LeftNode)[0]) -
                   gt->deltat_n * uleft));
    // energie cinetique
    LeftupwindVelocity(pNode)[2] =
        varlp->DualPhi(pNode)[2] +
        order2 * (0.5 * gradDualPhi0[2] *
                  (signleft * (varlp->XLagrange(pNode)[0] -
                               varlp->XLagrange(LeftNode)[0]) -
                   gt->deltat_n * uleft));
  }
}
// *********************************************
// Méthode A2
// *********************************************
void Remap::getRightAndLeftFluxMasseViaVol1(const int nbmat,
                                            const size_t pNodes) {
  // construction des mailles et faces associées pour recuperer les
  // flux de masses aux faces à gauche et à droite
#include "FacesLeftAndRight.h"
  // on prend moyenne les flux de masses (nbmat + imat)
  // des 4 faces verticales des 2 mailles à droite du noeud
  // ou
  // des 4 faces verticales des 2 mailles à gauche du noeud
  RightFluxMasse(pNodes) = 0.;
  LeftFluxMasse(pNodes) = 0.;
  for (int imat = 0; imat < nbmat; imat++) {
    RightFluxMassePartielle(pNodes)[imat] = 0;
    LeftFluxMassePartielle(pNodes)[imat] = 0;

    if (VerticalFaceOfNode(pNodes)[0] != -1) {
      RightFluxMassePartielle(pNodes)[imat] +=
          varlp->rLagrange(cfCell1) *
          (FluxFace1(cfCell1, fOfcfCell1)[imat] *
               varlp->outerFaceNormal(cfCell1, fOfcfCell1)[0] +
           FluxFace1(cfCell1, frOfcfCell1)[imat] *
               varlp->outerFaceNormal(cfCell1, frOfcfCell1)[0]);
      LeftFluxMassePartielle(pNodes)[imat] +=
          varlp->rLagrange(cbCell1) *
          (FluxFace1(cbCell1, fOfcbCell1)[imat] *
               varlp->outerFaceNormal(cbCell1, fOfcbCell1)[0] +
           FluxFace1(cbCell1, flOfcbCell1)[imat] *
               varlp->outerFaceNormal(cbCell1, flOfcbCell1)[0]);
    }

    if (VerticalFaceOfNode(pNodes)[1] != -1) {
      RightFluxMassePartielle(pNodes)[imat] +=
          varlp->rLagrange(cfCell2) *
          (FluxFace1(cfCell2, fOfcfCell2)[imat] *
               varlp->outerFaceNormal(cfCell2, fOfcfCell2)[0] +
           FluxFace1(cfCell2, frOfcfCell2)[imat] *
               varlp->outerFaceNormal(cfCell2, frOfcfCell2)[0]);
      LeftFluxMassePartielle(pNodes)[imat] +=
          varlp->rLagrange(cbCell2) *
          (FluxFace1(cbCell2, fOfcbCell2)[imat] *
               varlp->outerFaceNormal(cbCell2, fOfcbCell2)[0] +
           FluxFace1(cbCell2, flOfcbCell2)[imat] *
               varlp->outerFaceNormal(cbCell2, flOfcbCell2)[0]);
    }
    // if (pNodes == 300 || pNodes == 301 || pNodes == 302) {
    // std::cout << " H1 pNode " <<  pNodes << " left " << LeftFluxMasse(pNodes)
    // 	      << " FluxFace1(cfCell1, fOfcfCell1)[imat] " << FluxFace1(cfCell1,
    // fOfcfCell1)[imat]
    // 	      << " FluxFace1(cfCell1, frOfcfCell1)[imat] " << FluxFace1(cfCell1,
    // frOfcfCell1)[imat]
    // 	      << std::endl;
    // std::cout << " H1 pNode " <<  pNodes << " FluxFace1(cfCell2,
    // fOfcfCell2)[imat] "
    // 	      << FluxFace1(cfCell2, fOfcfCell2)[imat]
    // 	      << std::endl;
    // std::cout << " H1 pNode " <<  pNodes << " FluxFace1(cfCell2,
    // frOfcfCell2)[imat] "
    // 	      << FluxFace1(cfCell2, frOfcfCell2)[imat]
    // 	      << std::endl;
    // }
    if (nbfaces != 0) {
      RightFluxMassePartielle(pNodes)[imat] /= nbfaces;
      LeftFluxMassePartielle(pNodes)[imat] /= nbfaces;
    }

    RightFluxMasse(pNodes) += RightFluxMassePartielle(pNodes)[imat];
    LeftFluxMasse(pNodes) += LeftFluxMassePartielle(pNodes)[imat];
  }
}
void Remap::getRightAndLeftFluxMasseViaVol2(const int nbmat,
                                            const size_t pNodes) {
  // construction des mailles et faces associées pour recuperer les
  // flux de masses aux faces à gauche et à droite
#include "FacesLeftAndRight.h"
  // on prend moyenne les flux de masses (nbmat + imat)
  // des 4 faces verticales des 2 mailles à droite du noeud
  // ou
  // des 4 faces verticales des 2 mailles à gauche du noeud
  RightFluxMasse(pNodes) = 0.;
  LeftFluxMasse(pNodes) = 0.;
  for (int imat = 0; imat < nbmat; imat++) {
    RightFluxMassePartielle(pNodes)[imat] = 0;
    LeftFluxMassePartielle(pNodes)[imat] = 0;

    if (VerticalFaceOfNode(pNodes)[0] != -1) {
      RightFluxMassePartielle(pNodes)[imat] +=
          varlp->rLagrange(cfCell1) *
          (FluxFace2(cfCell1, fOfcfCell1)[imat] *
               varlp->outerFaceNormal(cfCell1, fOfcfCell1)[0] +
           FluxFace2(cfCell1, frOfcfCell1)[imat] *
               varlp->outerFaceNormal(cfCell1, frOfcfCell1)[0]);
      LeftFluxMassePartielle(pNodes)[imat] +=
          varlp->rLagrange(cbCell1) *
          (FluxFace2(cbCell1, fOfcbCell1)[imat] *
               varlp->outerFaceNormal(cbCell1, fOfcbCell1)[0] +
           FluxFace2(cbCell1, flOfcbCell1)[imat] *
               varlp->outerFaceNormal(cbCell1, flOfcbCell1)[0]);
    }

    if (VerticalFaceOfNode(pNodes)[1] != -1) {
      RightFluxMassePartielle(pNodes)[imat] +=
          varlp->rLagrange(cfCell2) *
          (FluxFace2(cfCell2, fOfcfCell2)[imat] *
               varlp->outerFaceNormal(cfCell2, fOfcfCell2)[0] +
           FluxFace2(cfCell2, frOfcfCell2)[imat] *
               varlp->outerFaceNormal(cfCell2, frOfcfCell2)[0]);
      LeftFluxMassePartielle(pNodes)[imat] +=
          varlp->rLagrange(cbCell2) *
          (FluxFace2(cbCell2, fOfcbCell2)[imat] *
               varlp->outerFaceNormal(cbCell2, fOfcbCell2)[0] +
           FluxFace2(cbCell2, flOfcbCell2)[imat] *
               varlp->outerFaceNormal(cbCell2, flOfcbCell2)[0]);
    }
    // if (pNodes == 300 || pNodes == 301 || pNodes == 302) {
    // std::cout << " H2 pNode " <<  pNodes << " left " << LeftFluxMasse(pNodes)
    // 	      << " FluxFace2(cfCell1, fOfcfCell1)[imat] " << FluxFace2(cfCell1,
    // fOfcfCell1)[imat]
    // 	      << " FluxFace2(cfCell1, frOfcfCell1)[imat] " << FluxFace2(cfCell1,
    // frOfcfCell1)[imat]
    // 	      << std::endl;
    // std::cout << " H1 pNode " <<  pNodes << " FluxFace2(cfCell2,
    // fOfcfCell2)[imat] "
    // 	      << FluxFace2(cfCell2, fOfcfCell2)[imat]
    // 	      << std::endl;
    // std::cout << " H1 pNode " <<  pNodes << " FluxFace2(cfCell2,
    // frOfcfCell2)[imat] "
    // 	      << FluxFace2(cfCell2, frOfcfCell2)[imat]
    // 	      << std::endl;
    // }
    if (nbfaces != 0) {
      RightFluxMassePartielle(pNodes)[imat] /= nbfaces;
      LeftFluxMassePartielle(pNodes)[imat] /= nbfaces;
    }

    RightFluxMasse(pNodes) += RightFluxMassePartielle(pNodes)[imat];
    LeftFluxMasse(pNodes) += LeftFluxMassePartielle(pNodes)[imat];
  }
}

void Remap::getTopAndBottomFluxMasseViaVol1(const int nbmat,
                                            const size_t pNodes) {
  // construction des mailles et faces associées pour recuperer les
  // flux de masses aux faces dessus et dessous
#include "FacesTopAndBottom.h"
  // on prend moyenne les flux de masses (nbmat + imat)
  // des 4 faces verticales des 2 mailles à droite du noeud
  // ou
  // des 4 faces verticales des 2 mailles à gauche du noeud

  TopFluxMasse(pNodes) = 0.;
  BottomFluxMasse(pNodes) = 0;
  for (int imat = 0; imat < nbmat; imat++) {
    TopFluxMassePartielle(pNodes)[imat] = 0.;
    BottomFluxMassePartielle(pNodes)[imat] = 0;

    if (HorizontalFaceOfNode(pNodes)[0] != -1) {
      TopFluxMassePartielle(pNodes)[imat] +=
          varlp->rLagrange(cbCell1) *
          (FluxFace1(cbCell1, fOfcbCell1)[imat] *
               varlp->outerFaceNormal(cbCell1, fOfcbCell1)[1] +
           FluxFace1(cbCell1, ftOfcbCell1)[imat] *
               varlp->outerFaceNormal(cbCell1, ftOfcbCell1)[1]);
      BottomFluxMassePartielle(pNodes)[imat] +=
          varlp->rLagrange(cfCell1) *
          (FluxFace1(cfCell1, fOfcfCell1)[imat] *
               varlp->outerFaceNormal(cfCell1, fOfcfCell1)[1] +
           FluxFace1(cfCell1, fbOfcfCell1)[imat] *
               varlp->outerFaceNormal(cfCell1, fbOfcfCell1)[1]);
    }
    if (HorizontalFaceOfNode(pNodes)[1] != -1) {
      TopFluxMassePartielle(pNodes)[imat] +=
          varlp->rLagrange(cbCell2) *
          (FluxFace1(cbCell2, fOfcbCell2)[imat] *
               varlp->outerFaceNormal(cbCell2, fOfcbCell2)[1] +
           FluxFace1(cbCell2, ftOfcbCell2)[imat] *
               varlp->outerFaceNormal(cbCell2, ftOfcbCell2)[1]);
      BottomFluxMassePartielle(pNodes)[imat] +=
          varlp->rLagrange(cfCell2) *
          (FluxFace1(cfCell2, fOfcfCell2)[imat] *
               varlp->outerFaceNormal(cfCell2, fOfcfCell2)[1] +
           FluxFace1(cfCell2, fbOfcfCell2)[imat] *
               varlp->outerFaceNormal(cfCell2, fbOfcfCell2)[1]);
    }
    if (nbfaces != 0) {
      TopFluxMassePartielle(pNodes)[imat] /= nbfaces;
      BottomFluxMassePartielle(pNodes)[imat] /= nbfaces;
    }

    TopFluxMasse(pNodes) += TopFluxMassePartielle(pNodes)[imat];
    BottomFluxMasse(pNodes) += BottomFluxMassePartielle(pNodes)[imat];
  }
}
void Remap::getTopAndBottomFluxMasseViaVol2(const int nbmat,
                                            const size_t pNodes) {
  // construction des mailles et faces associées pour recuperer les
  // flux de masses aux faces dessus et dessous
#include "FacesTopAndBottom.h"
  // on prend moyenne les flux de masses (nbmat + imat)
  // des 4 faces verticales des 2 mailles à droite du noeud
  // ou
  // des 4 faces verticales des 2 mailles à gauche du noeud
  TopFluxMasse(pNodes) = 0.;
  BottomFluxMasse(pNodes) = 0;
  for (int imat = 0; imat < nbmat; imat++) {
    TopFluxMassePartielle(pNodes)[imat] = 0.;
    BottomFluxMassePartielle(pNodes)[imat] = 0;

    if (HorizontalFaceOfNode(pNodes)[0] != -1) {
      TopFluxMassePartielle(pNodes)[imat] +=
          varlp->rLagrange(cbCell1) *
          (FluxFace2(cbCell1, fOfcbCell1)[imat] *
               varlp->outerFaceNormal(cbCell1, fOfcbCell1)[1] +
           FluxFace2(cbCell1, ftOfcbCell1)[imat] *
               varlp->outerFaceNormal(cbCell1, ftOfcbCell1)[1]);
      BottomFluxMassePartielle(pNodes)[imat] +=
          varlp->rLagrange(cfCell1) *
          (FluxFace2(cfCell1, fOfcfCell1)[imat] *
               varlp->outerFaceNormal(cfCell1, fOfcfCell1)[1] +
           FluxFace2(cfCell1, fbOfcfCell1)[imat] *
               varlp->outerFaceNormal(cfCell1, fbOfcfCell1)[1]);
    }
    if (HorizontalFaceOfNode(pNodes)[1] != -1) {
      TopFluxMassePartielle(pNodes)[imat] +=
          varlp->rLagrange(cbCell2) *
          (FluxFace2(cbCell2, fOfcbCell2)[imat] *
               varlp->outerFaceNormal(cbCell2, fOfcbCell2)[1] +
           FluxFace2(cbCell2, ftOfcbCell2)[imat] *
               varlp->outerFaceNormal(cbCell2, ftOfcbCell2)[1]);
      BottomFluxMassePartielle(pNodes)[imat] +=
          varlp->rLagrange(cfCell2) *
          (FluxFace2(cfCell2, fOfcfCell2)[imat] *
               varlp->outerFaceNormal(cfCell2, fOfcfCell2)[1] +
           FluxFace2(cfCell2, fbOfcfCell2)[imat] *
               varlp->outerFaceNormal(cfCell2, fbOfcfCell2)[1]);
    }
    if (nbfaces != 0) {
      TopFluxMassePartielle(pNodes)[imat] /= nbfaces;
      BottomFluxMassePartielle(pNodes)[imat] /= nbfaces;
    }

    TopFluxMasse(pNodes) += TopFluxMassePartielle(pNodes)[imat];
    BottomFluxMasse(pNodes) += BottomFluxMassePartielle(pNodes)[imat];
  }
}
// *********************************************
// Méthode PB
// *********************************************
void Remap::getRightAndLeftFluxMassePB1(const int nbmat, const size_t pNodes) {
  // construction des mailles et faces associées pour recuperer les
  // flux de masses aux faces à gauche et à droite
#include "FacesLeftAndRight.h"
  // on prend moyenne les flux de masses (nbmat + imat)
  // des 4 faces verticales des 2 mailles à droite du noeud
  // ou
  // des 4 faces verticales des 2 mailles à gauche du noeud
  RightFluxMasse(pNodes) = 0.;
  LeftFluxMasse(pNodes) = 0.;
  for (int imat = 0; imat < nbmat; imat++) {
    RightFluxMassePartielle(pNodes)[imat] = 0;
    LeftFluxMassePartielle(pNodes)[imat] = 0;
    int nbcell(0);

    if (VerticalFaceOfNode(pNodes)[0] != -1) {
      RightFluxMassePartielle(pNodes)[imat] +=
          DualphiFlux1(cfCell1)[nbmat + imat];
      LeftFluxMassePartielle(pNodes)[imat] +=
          DualphiFlux1(cbCell1)[nbmat + imat];
      nbcell++;
    }

    if (VerticalFaceOfNode(pNodes)[1] != -1) {
      RightFluxMassePartielle(pNodes)[imat] +=
          DualphiFlux1(cfCell2)[nbmat + imat];
      LeftFluxMassePartielle(pNodes)[imat] +=
          DualphiFlux1(cbCell2)[nbmat + imat];
      nbcell++;
    }

    if (nbcell != 0) {
      RightFluxMassePartielle(pNodes)[imat] /= nbcell;
      LeftFluxMassePartielle(pNodes)[imat] /= nbcell;
    }

    RightFluxMasse(pNodes) += RightFluxMassePartielle(pNodes)[imat];
    LeftFluxMasse(pNodes) += LeftFluxMassePartielle(pNodes)[imat];

    // if (pNodes == 300 || pNodes == 301 || pNodes == 302) {
    // std::cout << " H1 pNode " <<  pNodes << " left " << LeftFluxMasse(pNodes)
    // 	      << " FluxFace1(cfCell1, fOfcfCell1)[nbmat+imat] " <<
    // FluxFace1(cfCell1, fOfcfCell1)[nbmat+imat]
    // 	      << " FluxFace1(cfCell1, frOfcfCell1)[nbmat+imat] " <<
    // FluxFace1(cfCell1, frOfcfCell1)[nbmat+imat]
    // 	      << std::endl;
    // std::cout << " H1 pNode " <<  pNodes << " FluxFace1(cfCell2,
    // fOfcfCell2)[nbmat+imat] "
    // 	      << FluxFace1(cfCell2, fOfcfCell2)[nbmat+imat]
    // 	      << std::endl;
    // std::cout << " H1 pNode " <<  pNodes << " FluxFace1(cfCell2,
    // frOfcfCell2)[nbmat+imat] "
    // 	      << FluxFace1(cfCell2, frOfcfCell2)[nbmat+imat]
    // 	      << std::endl;
    // }
  }
}
void Remap::getRightAndLeftFluxMassePB2(const int nbmat, const size_t pNodes) {
  // construction des mailles et faces associées pour recuperer les
  // flux de masses aux faces à gauche et à droite
#include "FacesLeftAndRight.h"
  // on prend moyenne les flux de masses (nbmat + imat)
  // des 4 faces verticales des 2 mailles à droite du noeud
  // ou
  // des 4 faces verticales des 2 mailles à gauche du noeud
  RightFluxMasse(pNodes) = 0.;
  LeftFluxMasse(pNodes) = 0.;
  for (int imat = 0; imat < nbmat; imat++) {
    RightFluxMassePartielle(pNodes)[imat] = 0;
    LeftFluxMassePartielle(pNodes)[imat] = 0;
    int nbcell(0);

    if (VerticalFaceOfNode(pNodes)[0] != -1) {
      RightFluxMassePartielle(pNodes)[imat] +=
          DualphiFlux2(cfCell1)[nbmat + imat];
      LeftFluxMassePartielle(pNodes)[imat] +=
          DualphiFlux2(cbCell1)[nbmat + imat];
      nbcell++;
    }

    if (VerticalFaceOfNode(pNodes)[1] != -1) {
      RightFluxMassePartielle(pNodes)[imat] +=
          DualphiFlux2(cfCell2)[nbmat + imat];
      LeftFluxMassePartielle(pNodes)[imat] +=
          DualphiFlux2(cbCell2)[nbmat + imat];
      nbcell++;
    }
    // if (pNodes == 300 || pNodes == 301 || pNodes == 302) {
    // std::cout << " H2 pNode " <<  pNodes << " left " << LeftFluxMasse(pNodes)
    // 	      << " FluxFace2(cfCell1, fOfcfCell1)[nbmat+imat] " <<
    // FluxFace2(cfCell1, fOfcfCell1)[nbmat+imat]
    // 	      << " FluxFace2(cfCell1, frOfcfCell1)[nbmat+imat] " <<
    // FluxFace2(cfCell1, frOfcfCell1)[nbmat+imat]
    // 	      << std::endl;
    // std::cout << " H1 pNode " <<  pNodes << " FluxFace2(cfCell2,
    // fOfcfCell2)[nbmat+imat] "
    // 	      << FluxFace2(cfCell2, fOfcfCell2)[nbmat+imat]
    // 	      << std::endl;
    // std::cout << " H1 pNode " <<  pNodes << " FluxFace2(cfCell2,
    // frOfcfCell2)[nbmat+imat] "
    // 	      << FluxFace2(cfCell2, frOfcfCell2)[nbmat+imat]
    // 	      << std::endl;
    // }
    if (nbcell != 0) {
      RightFluxMassePartielle(pNodes)[imat] /= nbcell;
      LeftFluxMassePartielle(pNodes)[imat] /= nbcell;
    }

    RightFluxMasse(pNodes) += RightFluxMassePartielle(pNodes)[imat];
    LeftFluxMasse(pNodes) += LeftFluxMassePartielle(pNodes)[imat];
  }
}

void Remap::getTopAndBottomFluxMassePB1(const int nbmat, const size_t pNodes) {
  // construction des mailles et faces associées pour recuperer les
  // flux de masses aux faces dessus et dessous
#include "FacesTopAndBottom.h"
  // on prend moyenne les flux de masses (nbmat + imat)
  // des 4 faces verticales des 2 mailles à droite du noeud
  // ou
  // des 4 faces verticales des 2 mailles à gauche du noeud

  TopFluxMasse(pNodes) = 0.;
  BottomFluxMasse(pNodes) = 0;
  for (int imat = 0; imat < nbmat; imat++) {
    TopFluxMassePartielle(pNodes)[imat] = 0.;
    BottomFluxMassePartielle(pNodes)[imat] = 0;
    int nbcell(0);

    if (HorizontalFaceOfNode(pNodes)[0] != -1) {
      TopFluxMassePartielle(pNodes)[imat] +=
          DualphiFlux1(cbCell1)[nbmat + imat];
      BottomFluxMassePartielle(pNodes)[imat] +=
          DualphiFlux1(cfCell1)[nbmat + imat];
      nbcell++;
    }
    if (HorizontalFaceOfNode(pNodes)[1] != -1) {
      TopFluxMassePartielle(pNodes)[imat] +=
          DualphiFlux1(cbCell2)[nbmat + imat];
      BottomFluxMassePartielle(pNodes)[imat] +=
          DualphiFlux1(cfCell2)[nbmat + imat];
      nbcell++;
    }
    if (nbcell != 0) {
      TopFluxMassePartielle(pNodes)[imat] /= nbcell;
      BottomFluxMassePartielle(pNodes)[imat] /= nbcell;
    }

    TopFluxMasse(pNodes) += TopFluxMassePartielle(pNodes)[imat];
    BottomFluxMasse(pNodes) += BottomFluxMassePartielle(pNodes)[imat];
  }
}
void Remap::getTopAndBottomFluxMassePB2(const int nbmat, const size_t pNodes) {
  // construction des mailles et faces associées pour recuperer les
  // flux de masses aux faces dessus et dessous
#include "FacesTopAndBottom.h"
  // on prend moyenne les flux de masses (nbmat + imat)
  // des 4 faces verticales des 2 mailles à droite du noeud
  // ou
  // des 4 faces verticales des 2 mailles à gauche du noeud
  TopFluxMasse(pNodes) = 0.;
  BottomFluxMasse(pNodes) = 0;
  for (int imat = 0; imat < nbmat; imat++) {
    TopFluxMassePartielle(pNodes)[imat] = 0.;
    BottomFluxMassePartielle(pNodes)[imat] = 0;
    int nbcell(0);

    if (HorizontalFaceOfNode(pNodes)[0] != -1) {
      TopFluxMassePartielle(pNodes)[imat] +=
          DualphiFlux2(cbCell1)[nbmat + imat];
      BottomFluxMassePartielle(pNodes)[imat] +=
          DualphiFlux2(cfCell1)[nbmat + imat];
      nbcell++;
    }
    if (HorizontalFaceOfNode(pNodes)[1] != -1) {
      TopFluxMassePartielle(pNodes)[imat] +=
          DualphiFlux2(cbCell2)[nbmat + imat];
      BottomFluxMassePartielle(pNodes)[imat] +=
          DualphiFlux2(cfCell2)[nbmat + imat];
      nbcell++;
    }
    if (nbcell != 0) {
      TopFluxMassePartielle(pNodes)[imat] /= nbcell;
      BottomFluxMassePartielle(pNodes)[imat] /= nbcell;
    }

    TopFluxMasse(pNodes) += TopFluxMassePartielle(pNodes)[imat];
    BottomFluxMasse(pNodes) += BottomFluxMassePartielle(pNodes)[imat];
  }
}
