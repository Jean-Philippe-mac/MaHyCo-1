#ifndef EUCCLHYDREMAP_H
#define EUCCLHYDREMAP_H

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
#include "../includes/Sortie.h"
#include "../includes/VariablesLagRemap.h"
#include "../initialisations/Init.h"
#include "../particle_scheme/SchemaParticules.h"
#include "../remap/Remap.h"
#include "mesh/CartesianMesh2D.h"  // for CartesianMesh2D, CartesianM...
#include "mesh/MeshGeometry.h"     // for MeshGeometry
#include "mesh/PvdFileWriter2D.h"  // for PvdFileWriter2D
#include "types/Types.h"           // for RealArray1D, RealArray2D
#include "utils/Timer.h"           // for Timer

/*---------------------------------------*/
/*---------------------------------------*/
using namespace nablalib;

class Eucclhyd {
 public:
  struct interval {
    double inf, sup;
  };

  // private:
 public:
  CartesianMesh2D* mesh;
  optionschemalib::OptionsSchema::Options* options;
  sortielib::Sortie::SortieVariables* so;
  castestlib::CasTest::Test* test;
  particleslib::SchemaParticules* particules;
  conditionslimiteslib::ConditionsLimites::Cdl* cdl;
  limiteurslib::LimiteursClass::Limiteurs* limiteurs;
  eoslib::EquationDetat* eos;
  cstmeshlib::ConstantesMaillagesClass::ConstantesMaillages* cstmesh;
  gesttempslib::GestionTempsClass::GestTemps* gt;
  variableslagremaplib::VariablesLagRemap* varlp;
  initlib::Initialisations* init;
  Remap* remap;
  PvdFileWriter2D writer;
  PvdFileWriter2D writer_particle;
  int nbNodes, nbCells, nbFaces, nbCellsOfNode, nbNodesOfCell, nbNodesOfFace,
      nbCellsOfFace;

  // Global Variables
  int n, nbCalls;
  double lastDump;
  double m_global_total_energy_L, m_global_total_energy_T,
      m_global_total_energy_0;
  double m_total_masse_L, m_total_masse_T, m_total_masse_0;

  // cells a debuguer
  int dbgcell1 = -389;
  int dbgcell2 = -126;
  int dbgcell3 = -156;
  int face_debug1 = -2033;
  int face_debug2 = -410;
  int test_debug = 1;

  // coordonnees
  Kokkos::View<RealArray1D<dim>*> m_node_coord;
  Kokkos::View<RealArray1D<dim>*> m_cell_coord;
  Kokkos::View<double*> m_cell_coord_x;
  Kokkos::View<double*> m_cell_coord_y;
  // volume
  Kokkos::View<double*> m_euler_volume;
  Kokkos::View<RealArray1D<nbmatmax>*> m_lagrange_volume;
  // densite
  Kokkos::View<double*> m_density_n;
  Kokkos::View<double*> m_density_nplus1;
  Kokkos::View<RealArray1D<nbmatmax>*> m_density_env_n;
  Kokkos::View<RealArray1D<nbmatmax>*> m_density_env_nplus1;
  // masse
  Kokkos::View<double*> m_cell_mass;
  Kokkos::View<RealArray1D<nbmatmax>*> m_cell_mass_env;
  // masse aux noeuds
  // pression 
  Kokkos::View<double*> m_pressure;
  Kokkos::View<RealArray1D<nbmatmax>*> m_pressure_env;
  Kokkos::View<double*> m_pressure_env1;
  Kokkos::View<double*> m_pressure_env2;
  Kokkos::View<double*> m_pressure_env3;
  // energie interne
  Kokkos::View<double*> m_internal_energy_n;
  Kokkos::View<double*> m_internal_energy_nplus1;
  Kokkos::View<RealArray1D<nbmatmax>*> m_internal_energy_env_n;
  Kokkos::View<RealArray1D<nbmatmax>*> m_internal_energy_env_nplus1;
  Kokkos::View<RealArray1D<nbmatmax>*> m_dpde_env;
  // vitesses
  Kokkos::View<RealArray1D<dim>*> m_node_velocity_n;
  Kokkos::View<RealArray1D<dim>*> m_node_velocity_nplus1;
  // vitesse aux mailles
  Kokkos::View<RealArray1D<dim>*> m_cell_velocity_n;
  Kokkos::View<RealArray1D<dim>*> m_cell_velocity_nplus1;
  // vitesse du son
  Kokkos::View<double*> m_speed_velocity;
  Kokkos::View<RealArray1D<nbmatmax>*> m_speed_velocity_env;
  // fraction volumique
  Kokkos::View<RealArray1D<nbmatmax>*> m_fracvol_env;
  Kokkos::View<double*> m_fracvol_env1;
  Kokkos::View<double*> m_fracvol_env2;
  Kokkos::View<double*> m_fracvol_env3;
  Kokkos::View<RealArray1D<nbmatmax>*> m_node_fracvol;
  // fraction massique
  Kokkos::View<RealArray1D<nbmatmax>*> m_mass_fraction_env;
  // sorties interface
  Kokkos::View<double*> m_interface12;
  Kokkos::View<double*> m_interface13;
  Kokkos::View<double*> m_interface23;
  // sortie energie et masse globales
  Kokkos::View<double*> m_total_energy_0;
  Kokkos::View<double*> m_total_energy_T;
  Kokkos::View<double*> m_total_energy_L;
  Kokkos::View<double*> m_global_masse_0;
  Kokkos::View<double*> m_global_masse_T;
  Kokkos::View<double*> m_global_masse_L;
  
  
  
  Kokkos::View<RealArray1D<dim>**> m_lpc;
  Kokkos::View<RealArray1D<dim>**> m_nplus;
  Kokkos::View<RealArray1D<dim>**> m_nminus;
  Kokkos::View<double**> m_lplus;
  Kokkos::View<double**> m_lminus;
  Kokkos::View<double*> m_cell_perimeter;
  Kokkos::View<double*> m_cell_deltat;
  Kokkos::View<double*> m_x_cell_velocity;
  Kokkos::View<double*> m_y_cell_velocity;
  Kokkos::View<double**> m_pressure_extrap;
  Kokkos::View<RealArray1D<nbmatmax>**> m_pressure_env_extrap;
  Kokkos::View<RealArray1D<dim>**> m_cell_velocity_extrap;
  Kokkos::View<RealArray1D<dim>*> m_pressure_gradient;
  Kokkos::View<RealArray1D<dim>**> m_pressure_gradient_env;
  Kokkos::View<RealArray2D<dim, dim>*> m_velocity_gradient;
  Kokkos::View<RealArray1D<dim>**> m_node_force_n;
  Kokkos::View<RealArray1D<dim>**> m_node_force_nplus1;
  Kokkos::View<RealArray1D<dim>***> m_node_force_env_n;
  Kokkos::View<RealArray1D<dim>***> m_node_force_env_nplus1;
  Kokkos::View<RealArray1D<dim>*> m_node_G;
  Kokkos::View<RealArray2D<dim, dim>**> m_dissipation_matrix;
  Kokkos::View<RealArray2D<dim, dim>***> m_dissipation_matrix_env;
  Kokkos::View<RealArray2D<dim, dim>*> m_node_dissipation;

  utils::Timer global_timer;
  utils::Timer cpu_timer;
  utils::Timer io_timer;
  const size_t maxHardThread =
      Kokkos::DefaultExecutionSpace::impl_max_hardware_threads();

 public:
  Eucclhyd(optionschemalib::OptionsSchema::Options* aOptions,
           cstmeshlib::ConstantesMaillagesClass::ConstantesMaillages* acstmesh,
           gesttempslib::GestionTempsClass::GestTemps* agt,
           castestlib::CasTest::Test* aTest,
           conditionslimiteslib::ConditionsLimites::Cdl* aCdl,
           limiteurslib::LimiteursClass::Limiteurs* aLimiteurs,
           particleslib::SchemaParticules* aParticules,
           eoslib::EquationDetat* aEos, CartesianMesh2D* aCartesianMesh2D,
           variableslagremaplib::VariablesLagRemap* avarlp, Remap* aremap,
           initlib::Initialisations* ainit, 
           sortielib::Sortie::SortieVariables* asorties,
	   string output)
      : options(aOptions),
        cstmesh(acstmesh),
        gt(agt),
        test(aTest),
        cdl(aCdl),
        limiteurs(aLimiteurs),
        particules(aParticules),
        eos(aEos),
        mesh(aCartesianMesh2D),
        varlp(avarlp),
        remap(aremap),
        init(ainit),
        so(asorties),
        writer("EucclhydRemap", output),
        writer_particle("Particules", output),
        nbNodes(mesh->getNbNodes()),
        nbCells(mesh->getNbCells()),
        nbFaces(mesh->getNbFaces()),
        nbCellsOfNode(CartesianMesh2D::MaxNbCellsOfNode),
        nbNodesOfCell(CartesianMesh2D::MaxNbNodesOfCell),
        nbNodesOfFace(CartesianMesh2D::MaxNbNodesOfFace),
        nbCalls(0),
        lastDump(0.0),
        m_node_coord("node_coord", nbNodes),
        m_cell_coord("cell_coord", nbCells),
        m_cell_coord_x("cell_coord_x", nbCells),
        m_cell_coord_y("cell_coord_y", nbCells),
        m_lpc("lpc_n", nbNodes, nbCellsOfNode),
        m_nplus("nplus", nbNodes, nbCellsOfNode),
        m_nminus("nminus", nbNodes, nbCellsOfNode),
        m_lplus("lplus", nbNodes, nbCellsOfNode),
        m_lminus("lminus", nbNodes, nbCellsOfNode),
        m_pressure("p", nbCells),
        m_pressure_env1("pressure_env1", nbCells),
        m_pressure_env2("pressure_env2", nbCells),
        m_pressure_env3("pressure_env3", nbCells),
        m_pressure_env("pp", nbCells),
        m_cell_mass("cell_mass", nbCells),
        m_cell_mass_env("mp", nbCells),
        m_euler_volume("euler_volume", nbCells),
        m_mass_fraction_env("mass_fraction_env", nbCells),
        m_fracvol_env("fracvol_env", nbCells),
        m_node_fracvol("node_fracvol", nbNodes),
        m_fracvol_env1("fracvol_env1", nbCells),
        m_fracvol_env2("fracvol_env2", nbCells),
        m_fracvol_env3("fracvol_env3", nbCells),
        m_interface12("interface12", nbCells),
        m_interface23("interface23", nbCells),
        m_interface13("interface13", nbCells),
        m_lagrange_volume("lagrange_volume", nbCells),
        m_cell_perimeter("cell_perimeter", nbCells),
        m_speed_velocity("speed_velocity", nbCells),
        m_speed_velocity_env("speed_velocity_env", nbCells),
        m_cell_deltat("cell_deltat", nbCells),
        m_node_velocity_n("node_velocity_n", nbNodes),
        m_node_velocity_nplus1("node_velocity_nplus1", nbNodes),
        m_total_energy_0("total_energy_0", nbCells),
        m_total_energy_T("total_energy_T", nbCells),
        m_total_energy_L("total_energy_L", nbCells),
        m_global_masse_0("global_masse_0", nbCells),
        m_global_masse_T("global_masse_T", nbCells),
        m_global_masse_L("global_masse_L", nbCells),
        m_density_n("density_n", nbCells),
        m_density_nplus1("density_nplus1", nbCells),
        m_density_env_n("density_env_n", nbCells),
        m_density_env_nplus1("density_env_nplus1", nbCells),
        m_cell_velocity_n("cell_velocity_n", nbCells),
        m_cell_velocity_nplus1("cell_velocity_nplus1", nbCells),
        m_x_cell_velocity("x_cell_velocity", nbCells),
        m_y_cell_velocity("y_cell_velocity", nbCells),
        m_internal_energy_n("internal_energy_n", nbCells),
        m_internal_energy_nplus1("internal_energy_nplus1", nbCells),
        m_internal_energy_env_n("internal_energy_env_n", nbCells),
        m_internal_energy_env_nplus1("internal_energy_env_nplus1", nbCells),
        m_dpde_env("dpde_env", nbCells),
        m_pressure_extrap("pressure_extrap", nbCells, nbNodesOfCell),
        m_pressure_env_extrap("pressure_env_extrap", nbCells, nbNodesOfCell),
        m_cell_velocity_extrap("cell_velocity_extrap", nbCells, nbNodesOfCell),
        m_pressure_gradient("pressure_gradient", nbCells),
        m_pressure_gradient_env("pressure_gradient_env", nbCells, nbmatmax),
        m_velocity_gradient("velocity_gradient", nbCells),
        m_node_force_n("node_force_n", nbNodes, nbCellsOfNode),
        m_node_force_nplus1("node_force_nplus1", nbNodes, nbCellsOfNode),
        m_node_force_env_n("node_force_env_n", nbNodes, nbCellsOfNode,
                           nbmatmax),
        m_node_force_env_nplus1("node_force_env_nplus1", nbNodes, nbCellsOfNode,
                                nbmatmax),
        m_node_G("node_G", nbNodes),
        m_dissipation_matrix("dissipation_matrix", nbNodes, nbCellsOfNode),
        m_dissipation_matrix_env("dissipation_matrix_env", nbNodes,
                                 nbCellsOfNode, nbmatmax),
        m_node_dissipation("node_dissipation", nbNodes) {
    // Copy node coordinates
    const auto& gNodes = mesh->getGeometry()->getNodes();
    Kokkos::parallel_for(nbNodes, KOKKOS_LAMBDA(const int& rNodes) {
      m_node_coord(rNodes) = gNodes[rNodes];
    });
  }

 private:
  void computeBoundaryNodeVelocities() noexcept;
  RealArray1D<dim> nodeVelocityBoundaryCondition(int BC,
                                                 RealArray1D<dim> BCValue,
                                                 RealArray2D<dim, dim> Mp,
                                                 RealArray1D<dim> Gp);
  RealArray1D<dim> nodeVelocityBoundaryConditionCorner(
      int BC1, RealArray1D<dim> BCValue1, int BC2, RealArray1D<dim> BCValue2,
      RealArray2D<dim, dim> Mp, RealArray1D<dim> Gp);

  void setUpTimeLoopN() noexcept;

  void computeCornerNormal() noexcept;
  void computeEOS();
  void computePressionMoyenne() noexcept;
  void computeGradients() noexcept;
  void computeMass() noexcept;
  void computeDissipationMatrix() noexcept;
  void computem_cell_deltat() noexcept;
  void extrapolateValue() noexcept;
  void computeG() noexcept;
  void computeNodeDissipationMatrixAndG() noexcept;
  void computeNodeVelocity() noexcept;
  void computeFaceVelocity() noexcept;
  void computeLagrangePosition() noexcept;
  void computeSubCellForce() noexcept;
  void computeLagrangeVolumeAndCenterOfGravity() noexcept;
  void computeFacedeltaxLagrange() noexcept;
  void updateCellCenteredLagrangeVariables() noexcept;

  void remapCellcenteredVariable() noexcept;

  void switchalpharho_rho() noexcept;
  void switchrho_alpharho() noexcept;
  void PreparecellvariablesForParticles() noexcept;

  RealArray2D<2, 2> inverse(RealArray2D<2, 2> a);
  template <size_t N, size_t M>
  RealArray2D<N, M> tensProduct(RealArray1D<N> a, RealArray1D<M> b);
  double crossProduct2d(RealArray1D<2> a, RealArray1D<2> b);

  /**
   * Job dumpVariables called @2.0 in executeTimeLoopN method.
   * In variables: m_cell_coord_x, m_cell_coord_y, m_internal_energy_n, m, p,
   * m_density_n, t_n, v Out variables:
   */
  void dumpVariables() noexcept;

  /**
   * Job executeTimeLoopN called @4.0 in simulate method.
   * In variables: m_node_force_n, m_node_force_nplus1, G, M,
   * m_node_dissipation, ULagrange, Uremap1, Uremap2, m_cell_velocity_extrap,
   * m_cell_velocity_n, m_node_velocity_n, m_node_velocity_nplus1, X, XLagrange,
   * m_cell_coord, m_cell_coordLagrange, m_cell_coord_x, m_cell_coord_y, Xf,
   * bottomBC, bottomBCValue, c, cfl, deltat_n, deltat_nplus1, m_cell_deltat,
   * deltaxLagrange, eos, eosPerfectGas, m_internal_energy_n, faceLength,
   * faceNormal, faceNormalVelocity, gamma, gradPhi1, gradPhi2, gradPhiFace1,
   * gradPhiFace2, m_velocity_gradient, m_pressure_gradient, leftBC,
   * leftBCValue, lminus, m_lpc, lplus, m, nminus, nplus, outerFaceNormal, p,
   * m_pressure_extrap, m_cell_perimeter, phiFace1, phiFace2,
   * projectionLimiterId, projectionOrder, m_density_n, rightBC, rightBCValue,
   * spaceOrder, t_n, topBC, topBCValue, v, vLagrange, x_then_y_n Out variables:
   * m_node_force_nplus1, G, M, m_node_dissipation, ULagrange, Uremap1, Uremap2,
   * m_cell_velocity_extrap, m_cell_velocity_nplus1, m_node_velocity_nplus1,
   * XLagrange, m_cell_coordLagrange, c, deltat_nplus1, m_cell_deltat,
   * deltaxLagrange, m_internal_energy_nplus1, faceNormalVelocity, gradPhi1,
   * gradPhi2, gradPhiFace1, gradPhiFace2, m_velocity_gradient,
   * m_pressure_gradient, m, p, m_pressure_extrap, phiFace1, phiFace2,
   * m_density_nplus1, t_nplus1, vLagrange, x_then_y_nplus1
   */
  void executeTimeLoopN() noexcept;

  /**
   * Job computedeltat called @3.0 in executeTimeLoopN method.
   * In variables: cfl, deltat_n, m_cell_deltat
   * Out variables: deltat_nplus1
   */
  void computedeltat() noexcept;

  /**
   * Job updateTime called @4.0 in executeTimeLoopN method.
   * In variables: deltat_nplus1, t_n
   * Out variables: t_nplus1
   */
  void updateTime() noexcept;

 public:
  void simulate();
};

#include "Utiles-Impl.h"

#endif  // EUCCLHYDREMAP_H
