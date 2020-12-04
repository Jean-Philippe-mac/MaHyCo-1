#ifndef REMAP_H
#define REMAP_H

/*---------------------------------------*/
/*---------------------------------------*/

#include <stddef.h>  // for size_t

#include <Kokkos_Core.hpp>                // for KOKKOS_LAMBDA
#include <OpenMP/Kokkos_OpenMP_Exec.hpp>  // for OpenMP::impl_is_initialized
#include <algorithm>                      // for copy
#include <array>                          // for array
#include <string>                         // for allocator, string
#include <vector>                         // for vector

#include "../includes/CasTest.h"
#include "../includes/ConditionsLimites.h"
#include "../includes/Constantes.h"
#include "../includes/CstMesh.h"
#include "../includes/Eos.h"
#include "../includes/Freefunctions.h"
#include "../includes/GestionTemps.h"
#include "../includes/Limiteurs.h"
#include "../includes/Options.h"
#include "../includes/VariablesLagRemap.h"
#include "mesh/CartesianMesh2D.h"  // for CartesianMesh2D, CartesianM...
#include "mesh/MeshGeometry.h"     // for MeshGeometry
#include "mesh/PvdFileWriter2D.h"  // for PvdFileWriter2D
#include "types/Types.h"           // for RealArray1D, RealArray2D
#include "utils/Timer.h"           // for Timer
#include "types/MathFunctions.h"

/*---------------------------------------*/
/*---------------------------------------*/
using namespace nablalib;

class Remap {
 public:
  struct interval {
    double inf, sup;
  };

 private:
  CartesianMesh2D* mesh;
  optionschemalib::OptionsSchema::Options* options;
  cstmeshlib::ConstantesMaillagesClass::ConstantesMaillages* cstmesh;
  conditionslimiteslib::ConditionsLimites::Cdl* cdl;
  limiteurslib::LimiteursClass::Limiteurs* limiteurs;
  gesttempslib::GestionTempsClass::GestTemps* gt;
  variableslagremaplib::VariablesLagRemap* varlp;
  int nbNodes, nbCells, nbFaces, nbCellsOfNode, nbNodesOfCell, nbNodesOfFace,
      nbCellsOfFace, nbFacesOfCell;

  // cells a debuguer
  int dbgcell1 = -389;
  int dbgcell2 = -126;
  int dbgcell3 = -156;
  int face_debug1 = -2033;
  int face_debug2 = -410;
  int test_debug = 1;

  // Connectivity Variables
  Kokkos::View<double*> LfLagrange;
  Kokkos::View<double*> HvLagrange;
  Kokkos::View<RealArray1D<nbequamax>*> Uremap1;
  Kokkos::View<RealArray1D<nbequamax>*> UDualremap1;
  Kokkos::View<RealArray1D<nbequamax>*> gradPhiFace1;
  Kokkos::View<RealArray1D<nbequamax>*> gradPhiFace2;
  Kokkos::View<RealArray1D<nbequamax>*> gradPhi1;
  Kokkos::View<RealArray1D<nbequamax>*> gradPhi2;
  Kokkos::View<RealArray1D<nbequamax>*> phiFace1;
  Kokkos::View<RealArray1D<nbequamax>*> phiFace2;
  Kokkos::View<RealArray1D<nbequamax>*> DualphiFlux1;
  Kokkos::View<RealArray1D<nbequamax>*> DualphiFlux2;
  Kokkos::View<RealArray1D<nbequamax>*> Bidon1;
  Kokkos::View<RealArray1D<nbequamax>*> Bidon2;
  Kokkos::View<RealArray1D<nbequamax>*> deltaPhiFaceAv;
  Kokkos::View<RealArray1D<nbequamax>*> deltaPhiFaceAr;
  Kokkos::View<RealArray1D<nbequamax>**> FluxFace1;
  Kokkos::View<RealArray1D<nbequamax>**> FluxFace2;
  Kokkos::View<RealArray1D<nbmatmax>*> RightFluxMassePartielle;
  Kokkos::View<RealArray1D<nbmatmax>*> LeftFluxMassePartielle;
  Kokkos::View<RealArray1D<nbmatmax>*> TopFluxMassePartielle;
  Kokkos::View<RealArray1D<nbmatmax>*> BottomFluxMassePartielle;
  Kokkos::View<double*> RightFluxMasse;
  Kokkos::View<double*> LeftFluxMasse;
  Kokkos::View<double*> TopFluxMasse;
  Kokkos::View<double*> BottomFluxMasse;
  Kokkos::View<RealArray1D<2>*> VerticalFaceOfNode;
  Kokkos::View<RealArray1D<2>*> HorizontalFaceOfNode;
  Kokkos::View<RealArray1D<dimplus1>*> TopupwindVelocity;
  Kokkos::View<RealArray1D<dimplus1>*> BottomupwindVelocity;
  Kokkos::View<RealArray1D<dimplus1>*> RightupwindVelocity;
  Kokkos::View<RealArray1D<dimplus1>*> LeftupwindVelocity;
  Kokkos::View<RealArray1D<nbequamax>*> gradDualPhi1;
  Kokkos::View<RealArray1D<nbequamax>*> gradDualPhi2;

 public:
  Remap(optionschemalib::OptionsSchema::Options* aOptions,
        cstmeshlib::ConstantesMaillagesClass::ConstantesMaillages* acstmesh,
        gesttempslib::GestionTempsClass::GestTemps* agt,
        conditionslimiteslib::ConditionsLimites::Cdl* aCdl,
        limiteurslib::LimiteursClass::Limiteurs* aLimiteurs,
        CartesianMesh2D* aCartesianMesh2D,
        variableslagremaplib::VariablesLagRemap* avarlp)
      : options(aOptions),
        cstmesh(acstmesh),
        gt(agt),
        cdl(aCdl),
        limiteurs(aLimiteurs),
        mesh(aCartesianMesh2D),
        varlp(avarlp),
        nbNodes(mesh->getNbNodes()),
        nbCells(mesh->getNbCells()),
        nbFaces(mesh->getNbFaces()),
        nbCellsOfNode(CartesianMesh2D::MaxNbCellsOfNode),
        nbNodesOfCell(CartesianMesh2D::MaxNbNodesOfCell),
        nbNodesOfFace(CartesianMesh2D::MaxNbNodesOfFace),
        nbFacesOfCell(CartesianMesh2D::MaxNbFacesOfCell),
        LfLagrange("LfLagrange", nbFaces),
        HvLagrange("HvLagrange", nbCells),
        Uremap1("Uremap1", nbCells),
        UDualremap1("UDualremap1", nbNodes),
        gradPhiFace1("gradPhiFace1", nbFaces),
        gradPhiFace2("gradPhiFace2", nbFaces),
        gradPhi1("gradPhi1", nbCells),
        gradPhi2("gradPhi2", nbCells),
        gradDualPhi1("gradPhi1", nbNodes),
        gradDualPhi2("gradPhi2", nbNodes),
        phiFace1("phiFace1", nbFaces),
        phiFace2("phiFace2", nbFaces),
        DualphiFlux1("DualphiFlux1", nbCells),
        DualphiFlux2("DualphiFlux2", nbCells),
        Bidon1("Bidon1", nbCells),
        Bidon2("Bidon2", nbCells),
        FluxFace1("FluxFace1", nbCells, nbFacesOfCell),
        FluxFace2("FluxFace2", nbCells, nbFacesOfCell),
        deltaPhiFaceAv("deltaPhiFaceAv", nbCells),
        deltaPhiFaceAr("deltaPhiFaceAr", nbCells),
        RightFluxMassePartielle("FluxMassePartielle", nbNodes),
        LeftFluxMassePartielle("FluxMassePartielle", nbNodes),
        TopFluxMassePartielle("FluxMassePartielle", nbNodes),
        BottomFluxMassePartielle("FluxMassePartielle", nbNodes),
        RightFluxMasse("FluxMasse", nbNodes),
        LeftFluxMasse("FluxMasse", nbNodes),
        TopFluxMasse("FluxMasse", nbNodes),
        BottomFluxMasse("FluxMasse", nbNodes),
        VerticalFaceOfNode("VFaceOfNode", nbNodes),
        HorizontalFaceOfNode("HFaceOfNode", nbNodes),
        TopupwindVelocity("upwindVelocity", nbNodes),
        BottomupwindVelocity("upwindVelocity", nbNodes),
        RightupwindVelocity("upwindVelocity", nbNodes),
        LeftupwindVelocity("upwindVelocity", nbNodes) {}

  void computeGradPhiFace1() noexcept;
  void computeGradPhi1() noexcept;
  void computeUpwindFaceQuantitiesForProjection1() noexcept;
  void computeUremap1() noexcept;
  void computeDualUremap1() noexcept;

  void computeGradPhiFace2() noexcept;
  void computeGradPhi2() noexcept;
  void computeUpwindFaceQuantitiesForProjection2() noexcept;
  void computeUremap2() noexcept;
  void computeDualUremap2() noexcept;
  void FacesOfNode();

  template <size_t d>
  RealArray1D<d> computeRemapFlux(int projectionOrder,
                                  int projectionAvecPlateauPente,
                                  double face_normal_velocity,
                                  RealArray1D<dim> face_normal,
                                  double face_length, RealArray1D<d> phi_face,
                                  RealArray1D<dim> outer_face_normal,
                                  RealArray1D<dim> exy, double deltat_n);

  RealArray1D<nbequamax> computeBoundaryFluxes(int proj, int cCells,
                                               RealArray1D<dim> exy);

 private:
  int getLeftCells(const int cells);
  int getRightCells(const int cells);
  int getBottomCells(const int cells);
  int getTopCells(const int cells);
  int getLeftNode(const int node);
  int getRightNode(const int node);
  int getBottomNode(const int node);
  int getTopNode(const int node);

  void getRightAndLeftFluxMasse1(const int nbmat, const size_t pNodes);
  void getRightAndLeftFluxMasse2(const int nbmat, const size_t pNodes);
  void getTopAndBottomFluxMasse1(const int nbmat, const size_t pNodes);
  void getTopAndBottomFluxMasse2(const int nbmat, const size_t pNodes);
  void getRightAndLeftFluxMasseViaVol1(const int nbmat, const size_t pNodes);
  void getRightAndLeftFluxMasseViaVol2(const int nbmat, const size_t pNodes);
  void getTopAndBottomFluxMasseViaVol1(const int nbmat, const size_t pNodes);
  void getTopAndBottomFluxMasseViaVol2(const int nbmat, const size_t pNodes);
  void getRightAndLeftFluxMassePB1(const int nbmat, const size_t pNodes);
  void getRightAndLeftFluxMassePB2(const int nbmat, const size_t pNodes);
  void getTopAndBottomFluxMassePB1(const int nbmat, const size_t pNodes);
  void getTopAndBottomFluxMassePB2(const int nbmat, const size_t pNodes);

  double fluxLimiter(int projectionLimiterId, double r);
  double fluxLimiterG(int projectionLimiterId, double gradplus,
                       double gradmoins, double y0, double yplus, double ymoins,
                       double h0, double hplus, double hmoins);
  double computeY0(int projectionLimiterId, double y0, double yplus,
                   double ymoins, double h0, double hplus, double hmoins,
                   int type);
  double computexgxd(double y0, double yplus, double ymoins, double h0,
                     double y0plus, double y0moins, int type);
  double computeygyd(double y0, double yplus, double ymoins, double h0,
                     double y0plus, double y0moins, double grady, int type);
  double INTY(double X, double x0, double y0, double x1, double y1);

  void getTopUpwindVelocity(const size_t TopNode, const size_t pNode,
                            RealArray1D<nbequamax> gradDualPhiT,
                            RealArray1D<nbequamax> gradDualPhi);

  void getRightUpwindVelocity(const size_t RightNode, const size_t pNode,
                              RealArray1D<nbequamax> gradDualPhiR,
                              RealArray1D<nbequamax> gradDualPhi);

  void getBottomUpwindVelocity(const size_t BottomNode, const size_t pNode,
                               RealArray1D<nbequamax> gradDualPhiB,
                               RealArray1D<nbequamax> gradDualPhi);

  void getLeftUpwindVelocity(const size_t LeftNode, const size_t pNode,
                             RealArray1D<nbequamax> gradDualPhiL,
                             RealArray1D<nbequamax> gradDualPhi);

  template <size_t d>
  RealArray1D<d> computeDualHorizontalGradPhi(RealArray1D<d> gradphiplus,
                                              RealArray1D<d> gradphimoins,
                                              const size_t pNode);
  template <size_t d>
  RealArray1D<d> computeDualVerticalGradPhi(RealArray1D<d> gradphiplus,
                                            RealArray1D<d> gradphimoins,
                                            const size_t pNode);
  template <size_t d>
  RealArray1D<d> computeAndLimitGradPhi(
      int projectionLimiterId, RealArray1D<d> gradphiplus,
      RealArray1D<d> gradphimoins, RealArray1D<d> phi, RealArray1D<d> phiplus,
      RealArray1D<d> phimoins, double h0, double hplus, double hmoins);

  template <size_t d>
  void computeFluxPP(RealArray1D<d> gradphi, RealArray1D<d> phi,
                     RealArray1D<d> phiplus, RealArray1D<d> phimoins, double h0,
                     double hplus, double hmoins, double face_normal_velocity,
                     double deltat_n, int type, int cell, double flux_threhold,
                     int projectionPlateauPenteComplet,
                     double dual_normal_velocity, int calcul_flux_dual,
                     RealArray1D<d>* Flux, RealArray1D<d>* Flux_dual);

  template <size_t d>
  void computeFluxPPPure(RealArray1D<d> gradphi, RealArray1D<d> phi,
                         RealArray1D<d> phiplus, RealArray1D<d> phimoins,
                         double h0, double hplus, double hmoins,
                         double face_normal_velocity, double deltat_n, int type,
                         int cell, double flux_threhold,
                         int projectionPlateauPenteComplet,
                         double dual_normal_velocity, int calcul_flux_dual,
                         RealArray1D<d>* Flux, RealArray1D<d>* Flux_dual);
  template <size_t d>
  RealArray1D<d> computeUpwindFaceQuantities(
      RealArray1D<dim> face_normal, double face_normal_velocity, double delta_x,
      RealArray1D<dim> x_f, RealArray1D<d> phi_cb, RealArray1D<d> grad_phi_cb,
      RealArray1D<dim> x_cb, RealArray1D<d> phi_cf, RealArray1D<d> grad_phi_cf,
      RealArray1D<dim> x_cf);
  RealArray1D<dim> xThenYToDirection(bool x_then_y_);

  template <size_t d>
  RealArray1D<d> computeVecFluxOrdre3(
      RealArray1D<d> phimmm, RealArray1D<d> phimm, RealArray1D<d> phim,
      RealArray1D<d> phip, RealArray1D<d> phipp, RealArray1D<d> phippp,
      double hmmm, double hmm, double hm, double hp, double hpp, double hppp,
      double face_normal_velocity, double deltat_n);
  interval define_interval(double a, double b);
  interval intersection(interval I1, interval I2);
  double evaluate_grad(double hm, double h0, double hp, double ym, double y0,
                       double yp);
  double evaluate_ystar(double hmm, double hm, double hp, double hpp,
                        double ymm, double ym, double yp, double ypp,
                        double gradm, double gradp);
  double evaluate_fm(double x, double dx, double up, double du, double u6);
  double evaluate_fp(double x, double dx, double um, double du, double u6);
  double ComputeFluxOrdre3(double ymmm, double ymm, double ym, double yp,
                           double ypp, double yppp, double hmmm, double hmm,
                           double hm, double hp, double hpp, double hppp,
                           double v_dt);

 public:
};

#include "UtilesRemap-Impl.h"

#endif  // EUCCLHYDREMAP_H
