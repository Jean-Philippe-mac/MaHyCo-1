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
 * \file computeDualUremap1()
 * \brief phase de projection duale
 *   etape 1 - horizontale et verticale suivant le cas
 *   calcul des flux de masses duales a partir des flux de masses primales
 *   selon 3 methodes differentes (A1, A2, PB)
 *           getRightAndLeftFluxMasse...
 *     ou    getTopAndBottomFluxMasse...
 *   reconstruction de la vitesse ou l'energie cinetique a l'ordre 1 ou 2
 *           getLeftUpwindVelocity, getRightUpwindVelocity
 *     ou    getBottomUpwindVelocity, getTopUpwindVelocity
 *
 * \param  varlp->UDualLagrange
 * \return UDualremap1, varlp->DualPhi
 *******************************************************************************
 */
void Remap::computeDualUremap1() noexcept {
  if (varlp->x_then_y_n) {
    if (options->projectionOrder > 1) {
      // calcul des gradients de vitesses
      Kokkos::parallel_for(nbNodes, KOKKOS_LAMBDA(const size_t& pNode) {
        RealArray1D<nbequamax> grad_right = Uzero;
        RealArray1D<nbequamax> grad_left = Uzero;
        gradDualPhi1(pNode) =
            computeDualHorizontalGradPhi(grad_right, grad_left, pNode);
      });
    } else {
      Kokkos::parallel_for(nbNodes, KOKKOS_LAMBDA(const size_t& pNode) {
        gradDualPhi1(pNode) = Uzero;
      });
    }
    Kokkos::parallel_for(nbNodes, KOKKOS_LAMBDA(const size_t& pNode) {
      int nbmat = options->nbmat;
      if (options->methode_flux_masse == 0)
        getRightAndLeftFluxMasse1(nbmat, pNode);

      if (options->methode_flux_masse == 1)
        getRightAndLeftFluxMasseViaVol1(nbmat, pNode);

      if (options->methode_flux_masse == 2)
        getRightAndLeftFluxMassePB1(nbmat, pNode);

      UDualremap1(pNode)[3] = varlp->UDualLagrange(pNode)[3] +
                              LeftFluxMasse(pNode) - RightFluxMasse(pNode);

      if (options->projectionOrder >= 1) {
        // Rightvitesse = vitesse(pNode) si RightFluxMasse(pNode) > 0 et
        // vitesse(voisin de droite) sinon Leftvitesse = vitesse(voisin de
        // gauche) si LeftFluxMasse(pNode) > 0 et vitesse(pNode) sinon

        int LeftNode = mesh->getLeftNode(pNode);
        int RightNode = mesh->getRightNode(pNode);
        getLeftUpwindVelocity(LeftNode, pNode, gradDualPhi1(LeftNode),
                              gradDualPhi1(pNode));
        getRightUpwindVelocity(RightNode, pNode, gradDualPhi1(RightNode),
                               gradDualPhi1(pNode));

        UDualremap1(pNode)[0] =
            varlp->UDualLagrange(pNode)[0] +
            LeftFluxMasse(pNode) * LeftupwindVelocity(pNode)[0] -
            RightFluxMasse(pNode) * RightupwindVelocity(pNode)[0];

        UDualremap1(pNode)[1] =
            varlp->UDualLagrange(pNode)[1] +
            LeftFluxMasse(pNode) * LeftupwindVelocity(pNode)[1] -
            RightFluxMasse(pNode) * RightupwindVelocity(pNode)[1];
	
	// energie cinetique
        UDualremap1(pNode)[2] =
            varlp->UDualLagrange(pNode)[2] +
            LeftFluxMasse(pNode) * LeftupwindVelocity(pNode)[2] -
            RightFluxMasse(pNode) * RightupwindVelocity(pNode)[2];
      }
    });
  } else {
    // projection verticale
    if (options->projectionOrder > 1) {
      // calcul des gradients de vitesses
      Kokkos::parallel_for(nbNodes, KOKKOS_LAMBDA(const size_t& pNode) {
        RealArray1D<nbequamax> grad_up = Uzero;
        RealArray1D<nbequamax> grad_down = Uzero;
        gradDualPhi1(pNode) =
            computeDualVerticalGradPhi(grad_up, grad_down, pNode);
      });
    } else {
      Kokkos::parallel_for(nbNodes, KOKKOS_LAMBDA(const size_t& pNode) {
        gradDualPhi1(pNode) = Uzero;
      });
    }
    Kokkos::parallel_for(nbNodes, KOKKOS_LAMBDA(const size_t& pNode) {
      int nbmat = options->nbmat;
      if (options->methode_flux_masse == 0)
        getTopAndBottomFluxMasse1(nbmat, pNode);
      if (options->methode_flux_masse == 1)
        getTopAndBottomFluxMasseViaVol1(nbmat, pNode);
      if (options->methode_flux_masse == 2)
        getTopAndBottomFluxMassePB1(nbmat, pNode);
      
      UDualremap1(pNode)[3] = varlp->UDualLagrange(pNode)[3] +
                              BottomFluxMasse(pNode) - TopFluxMasse(pNode);
      
      if (options->projectionOrder >= 1) {
        // recherche de la vitesse du decentrement upwind
        // Topvitesse = vitesse(pNode) si TopFluxMasse(pNode) > 0 et
        // vitesse(voisin du haut) sinon Bottomvitesse = vitesse(voisin du
        // bas) si BottomFluxMasse(pNode) > 0 et vitesse(pNode) sinon

        int TopNode = mesh->getTopNode(pNode);
        int BottomNode = mesh->getBottomNode(pNode);
        getBottomUpwindVelocity(BottomNode, pNode, gradDualPhi1(BottomNode),
                                gradDualPhi1(pNode));
        getTopUpwindVelocity(TopNode, pNode, gradDualPhi1(TopNode),
                             gradDualPhi1(pNode));

        UDualremap1(pNode)[0] =
            varlp->UDualLagrange(pNode)[0] +
            BottomFluxMasse(pNode) * BottomupwindVelocity(pNode)[0] -
            TopFluxMasse(pNode) * TopupwindVelocity(pNode)[0];

        UDualremap1(pNode)[1] =
            varlp->UDualLagrange(pNode)[1] +
            BottomFluxMasse(pNode) * BottomupwindVelocity(pNode)[1] -
            TopFluxMasse(pNode) * TopupwindVelocity(pNode)[1];
	// energie cinetique
	UDualremap1(pNode)[2] =
            varlp->UDualLagrange(pNode)[2] +
            BottomFluxMasse(pNode) * BottomupwindVelocity(pNode)[2] -
            TopFluxMasse(pNode) * TopupwindVelocity(pNode)[2];
      }
    });
  }
  Kokkos::parallel_for(nbNodes, KOKKOS_LAMBDA(const size_t& pNode) {
    // vitesse x Y
    varlp->DualPhi(pNode)[0] = UDualremap1(pNode)[0] / UDualremap1(pNode)[3];
    varlp->DualPhi(pNode)[1] = UDualremap1(pNode)[1] / UDualremap1(pNode)[3];
    // masse nodale
    varlp->DualPhi(pNode)[3] = UDualremap1(pNode)[3];
    // energie cinétique
    varlp->DualPhi(pNode)[2] = UDualremap1(pNode)[2] / UDualremap1(pNode)[3];
  });
}
