#ifndef UTILESREMAP_IMPL_H
#define UTILESREMAP_IMPL_H

#include "Remap.h"          // for Remap, Remap::Opt...
//#include "types/ArrayOperations.h"
#include "types/MathFunctions.h"

template <size_t d>
RealArray1D<d> Remap::computeAndLimitGradPhi(
    int projectionLimiterId, RealArray1D<d> gradphiplus,
    RealArray1D<d> gradphimoins, RealArray1D<d> phi, RealArray1D<d> phiplus,
    RealArray1D<d> phimoins, double h0, double hplus, double hmoins) {
  RealArray1D<d> res;
  if (projectionLimiterId < limiteurs->minmodG) {
    // std::cout << " Passage gradient limite Classique " << std::endl;
    for (size_t i = 0; i < d; i++) {
      res[i] = (fluxLimiter(projectionLimiterId,
                            divideNoExcept(gradphiplus[i], gradphimoins[i])) *
                    gradphimoins[i] +
                fluxLimiter(projectionLimiterId,
                            divideNoExcept(gradphimoins[i], gradphiplus[i])) *
                    gradphiplus[i]) /
               2.0;
    }
    return res;
  } else {
    // std::cout << " Passage gradient limite Genéralisé " << std::endl;
    for (size_t i = 0; i < d; i++) {
      res[i] =
          fluxLimiterPP(projectionLimiterId, gradphiplus[i], gradphimoins[i],
                        phi[i], phiplus[i], phimoins[i], h0, hplus, hmoins);
    }
    return res;
  }
}
template <size_t d>
RealArray1D<d> Remap::computeVecFluxOrdre3(
    RealArray1D<d> phimmm, RealArray1D<d> phimm, RealArray1D<d> phim,
    RealArray1D<d> phip, RealArray1D<d> phipp, RealArray1D<d> phippp,
    double hmmm, double hmm, double hm, double hp, double hpp, double hppp,
    double face_normal_velocity, double deltat_n) {
  RealArray1D<d> res;
  double v_dt = face_normal_velocity * deltat_n;
  for (size_t i = 0; i < d; i++) {
    res[i] = ComputeFluxOrdre3(phimmm[i], phimm[i], phim[i], phip[i], phipp[i],
                               phippp[i], hmmm, hmm, hm, hp, hpp, hppp, v_dt);
  }
  return res;
}

template <size_t d>
RealArray1D<d> Remap::computeFluxPP(
    RealArray1D<d> gradphi, RealArray1D<d> phi, RealArray1D<d> phiplus,
    RealArray1D<d> phimoins, double h0, double hplus, double hmoins,
    double face_normal_velocity, double deltat_n, int type, int cell,
    double flux_threhold) {
  RealArray1D<d> Flux = Uzero;
  int nbmat = options->nbmat;
  double y0plus, y0moins, xd, xg, yd, yg;
  double flux1, flux2, flux3, flux1m, flux2m, flux3m;
  double partie_positive_v =
      0.5 * (face_normal_velocity + abs(face_normal_velocity)) * deltat_n;
  if (partie_positive_v == 0.) return Flux;
  int cas_PP = 0;
  for (size_t i = 0; i < nbmat; i++) {
    // calcul des seuils y0plus, y0moins pour cCells
    y0plus = computeY0(limiteurs->projectionLimiterId, phi[i], phiplus[i],
                       phimoins[i], h0, hplus, hmoins, 0);
    y0moins = computeY0(limiteurs->projectionLimiterId, phi[i], phiplus[i],
                        phimoins[i], h0, hplus, hmoins, 1);

    // calcul des points d'intersections xd,xg
    xg = computexgxd(phi[i], phiplus[i], phimoins[i], h0, y0plus, y0moins, 0);
    xd = computexgxd(phi[i], phiplus[i], phimoins[i], h0, y0plus, y0moins, 1);

    // calcul des valeurs sur ces points d'intersections
    // yg = computeygyd(phi[i], phiplus[i], phimoins[i], h0, gradphi[i], 0);
    // yd = computeygyd(phi[i], phiplus[i], phimoins[i], h0, gradphi[i], 1);

    yg = computeygyd(phi[i], phiplus[i], phimoins[i], h0, y0plus, y0moins,
                     gradphi[i], 0);
    yd = computeygyd(phi[i], phiplus[i], phimoins[i], h0, y0plus, y0moins,
                     gradphi[i], 1);

    // if (cell == -1009) {
    //   std::cout << " -------------" << std::endl;
    //   std::cout << " type = " << type << " I =" << i << " du mat 1"<<
    //   std::endl; std::cout << " y0moins " << y0moins << std::endl; std::cout
    //   << " y0plus " << y0plus << std::endl; std::cout << " phimoins " <<
    //   phimoins[i] << std::endl; std::cout << " phi " << phi[i] << std::endl;
    //   std::cout << " phiplus " << phiplus[i] << std::endl;
    //   std::cout << " h0 " << h0 << " donc " << h0/2. << std::endl;
    //   std::cout << " hmoins " << hmoins << std::endl;
    //   std::cout << " hplus " << hplus << std::endl;
    //   std::cout << " grady " << gradphi[i] << std::endl;
    //   std::cout << " xg " << xg << std::endl;
    //   std::cout << " xd " << xd << std::endl;
    //   std::cout << " yg " << yg << std::endl;
    //   std::cout << " yd " << yd << std::endl;
    //   std::cout << " partie_positive_v " << partie_positive_v << std::endl;
    //   cas_PP = 1;
    //  }

    if (type == 0)  // flux arriere ou en dessous de cCells, integration entre
                    // -h0/2. et -h0/2.+abs(face_normal_velocity)*deltat_n
    {
      // Flux1m : integrale -inf,  -h0/2.+partie_positive_v
      flux1m =
          INT2Y(-h0 / 2. + partie_positive_v, -h0 / 2., phimoins[i], xg, yg);
      // Flux1m : integrale -inf,  -h0/2.
      flux1 = INT2Y(-h0 / 2., -h0 / 2., phimoins[i], xg, yg);
      //
      // if (abs(flux1-flux1m) > flux_threhold && abs(xg) != abs(xd)) {
      //   std::cout << i << " Activation Plateau Pente  AR Flux -h0/2, xg non
      //   nul " << cell << " = " << abs(flux1-flux1m) << std::endl;

      //   std::cout << " xg " << xg << " et h0/2 " << h0/2. << std::endl;
      //   std::cout << " xd " << xd << " et h0/2 " << h0/2. << std::endl;
      //   cas_PP = 1;
      //  }
      //
      // Flux2m : integrale -inf,  -h0/2.+partie_positive_v
      flux2m = INT2Y(-h0 / 2. + partie_positive_v, xg, yg, xd, yd);
      // Flux2 : integrale -inf,  -h0/2.
      flux2 = INT2Y(-h0 / 2., xg, yg, xd, yd);
      //
      // Flux3m : integrale -inf,  -h0/2.+partie_positive_v
      flux3m = INT2Y(-h0 / 2. + partie_positive_v, xd, yd, h0 / 2., phiplus[i]);
      // Flux3 : integrale -inf,  -h0/2.
      flux3 = INT2Y(-h0 / 2., xd, yd, h0 / 2., phiplus[i]);
      //
      // if (abs(flux3-flux3m) > flux_threhold && abs(xg) != abs(xd)) {
      //   std::cout << i << " Activation Plateau Pente  AR Flux xd , h0/2, non
      //   nul " << cell << " = " << abs(flux3-flux3m) << std::endl; std::cout
      //   << " xg " << xg << " et h0/2 " << h0/2. << std::endl; std::cout << "
      //   xd " << xd << " et h0/2 " << h0/2. << std::endl; cas_PP = 1;
      // }
      //
      // integrale positive
      Flux[i] = MathFunctions::max(
          ((flux1m - flux1) + (flux2m - flux2) + (flux3m - flux3)), 0.);
      // formule 16
      if (((phiplus[i] - phi[i]) * (phimoins[i] - phi[i])) >= 0.)
        Flux[i] = phi[i] * partie_positive_v;
      //
      // if (cas_PP == 1) std::cout << " type 1 Flux1 " << flux1 << " Flux1m "
      // << flux1m << " Flux2 " << flux2
      //			    << " Flux2m " << flux2m << " Flux3 " <<
      // flux3 << " Flux3m " << flux3m << " soit " << Flux[i] << std::endl;
    } else if (type ==
               1)  // flux devant ou au dessus de cCells, integration entre
                   // h0/2.-abs(face_normal_velocity)*deltat_n et h0/2.
    {
      // Flux1 : integrale -inf,  h0/2.-partie_positive_v
      flux1 = INT2Y(h0 / 2. - partie_positive_v, -h0 / 2., phimoins[i], xg, yg);
      // Flux1m : integrale -inf,  -h0/2.
      flux1m = INT2Y(h0 / 2., -h0 / 2., phimoins[i], xg, yg);
      //
      // if (abs(flux1-flux1m) > flux_threhold && abs(xg) != abs(xd)) {
      //   std::cout << i << " Activation Plateau Pente  AV Flux -h0/2, xg non
      //   nul " << cell << " = " << abs(flux1-flux1m) << std::endl; std::cout
      //   << " xg " << xg << " et h0/2 " << h0/2. << std::endl; std::cout << "
      //   xd " << xd << " et h0/2 " << h0/2. << std::endl; cas_PP = 1;
      //  }
      //
      // Flux2 : integrale -inf,  h0/2.-partie_positive_v
      flux2 = INT2Y(h0 / 2. - partie_positive_v, xg, yg, xd, yd);
      // Flux2m : integrale -inf,  -h0/2.
      flux2m = INT2Y(h0 / 2., xg, yg, xd, yd);
      //
      // Flux3 : integrale -inf,  h0/2.-partie_positive_v
      flux3 = INT2Y(h0 / 2. - partie_positive_v, xd, yd, h0 / 2., phiplus[i]);
      // Flux3m : integrale -inf,  -h0/2.
      flux3m = INT2Y(h0 / 2., xd, yd, h0 / 2., phiplus[i]);
      // if (abs(flux3-flux3m) > flux_threhold && abs(xg) != abs(xd)) {
      //    std::cout << i << " Activation Plateau Pente  AV Flux xd , h0/2, non
      //    nul " << cell << " = " << abs(flux3-flux3m) << std::endl;
      //  std::cout << " xg " << xg << " et h0/2 " << h0/2. << std::endl;
      //  std::cout << " xd " << xd << " et h0/2 " << h0/2. << std::endl;
      //  cas_PP = 1;
      // }
      //
      // integrale positive
      Flux[i] = MathFunctions::max(
          ((flux1m - flux1) + (flux2m - flux2) + (flux3m - flux3)), 0.);
      // formule 16
      if (((phiplus[i] - phi[i]) * (phimoins[i] - phi[i])) >= 0.)
        Flux[i] = phi[i] * partie_positive_v;

      // if (cas_PP == 1) std::cout << " type 1 Flux1 " << flux1 << " Flux1m "
      // << flux1m << " Flux2 " << flux2
      //			    << " Flux2m " << flux2m << " Flux3 " <<
      // flux3 << " Flux3m " << flux3m << " soit " << Flux[i] << std::endl;
    }

  }
  // std::cout << " Flux " << Flux << std::endl;
  // les flux de masse, de quantité de mouvement et d'energie massique se
  // deduisent des flux de volumes
  double somme_flux_masse = 0.;
  double somme_flux_volume = 0.;
  for (size_t imat = 0; imat < nbmat; imat++) {
    Flux[nbmat + imat] =
        phi[nbmat + imat] * Flux[imat];  // flux de masse de imat
    Flux[2 * nbmat + imat] =
        phi[2 * nbmat + imat] *
        Flux[nbmat + imat];  // flux de masse energy de imat
    somme_flux_masse += Flux[nbmat + imat];
    somme_flux_volume += Flux[imat];
  }
  Flux[3 * nbmat] =
      phi[3 * nbmat] * somme_flux_masse;  // flux de quantité de mouvement x
  Flux[3 * nbmat + 1] =
      phi[3 * nbmat + 1] * somme_flux_masse;  // flux de quantité de mouvement y
  Flux[3 * nbmat + 2] =
      phi[3 * nbmat + 2] * somme_flux_masse;  // flux d'energie cinetique
  Flux[3 * nbmat + 3] =
    phi[3 * nbmat + 3] * somme_flux_volume; // flux pour la pseudo VNR

  return Flux;
}

template <size_t d>
RealArray1D<d> Remap::computeFluxPPPure(
    RealArray1D<d> gradphi, RealArray1D<d> phi, RealArray1D<d> phiplus,
    RealArray1D<d> phimoins, double h0, double hplus, double hmoins,
    double face_normal_velocity, double deltat_n, int type, int cell,
    double flux_threhold) {
  RealArray1D<d> Flux = Uzero;
  int nbmat = options->nbmat;
  double y0plus, y0moins, xd, xg, yd, yg;
  double flux1, flux2, flux3, flux1m, flux2m, flux3m;
  double partie_positive_v =
      0.5 * (face_normal_velocity + abs(face_normal_velocity)) * deltat_n;
  if (partie_positive_v == 0.) return Flux;
  int cas_PP = 0;
  // on ne fait que la projection des volumes et masses
  for (size_t i = 0; i < 2 * nbmat; i++) {
    // calcul des seuils y0plus, y0moins pour cCells
    y0plus = computeY0(limiteurs->projectionLimiterIdPure, phi[i], phiplus[i],
                       phimoins[i], h0, hplus, hmoins, 0);
    y0moins = computeY0(limiteurs->projectionLimiterIdPure, phi[i], phiplus[i],
                        phimoins[i], h0, hplus, hmoins, 1);

    // calcul des points d'intersections xd,xg
    xg = computexgxd(phi[i], phiplus[i], phimoins[i], h0, y0plus, y0moins, 0);
    xd = computexgxd(phi[i], phiplus[i], phimoins[i], h0, y0plus, y0moins, 1);

    // calcul des valeurs sur ces points d'intersections
    // yg = computeygyd(phi[i], phiplus[i], phimoins[i], h0, gradphi[i], 0);
    // yd = computeygyd(phi[i], phiplus[i], phimoins[i], h0, gradphi[i], 1);

    yg = computeygyd(phi[i], phiplus[i], phimoins[i], h0, y0plus, y0moins,
                     gradphi[i], 0);
    yd = computeygyd(phi[i], phiplus[i], phimoins[i], h0, y0plus, y0moins,
                     gradphi[i], 1);

    // if (cell == -1009) {
    //   std::cout << " -------------" << std::endl;
    //   std::cout << " type = " << type << " I =" << i << " du mat 1"<<
    //   std::endl; std::cout << " y0moins " << y0moins << std::endl; std::cout
    //   << " y0plus " << y0plus << std::endl; std::cout << " phimoins " <<
    //   phimoins[i] << std::endl; std::cout << " phi " << phi[i] << std::endl;
    //   std::cout << " phiplus " << phiplus[i] << std::endl;
    //   std::cout << " h0 " << h0 << " donc " << h0/2. << std::endl;
    //   std::cout << " hmoins " << hmoins << std::endl;
    //   std::cout << " hplus " << hplus << std::endl;
    //   std::cout << " grady " << gradphi[i] << std::endl;
    //   std::cout << " xg " << xg << std::endl;
    //   std::cout << " xd " << xd << std::endl;
    //   std::cout << " yg " << yg << std::endl;
    //   std::cout << " yd " << yd << std::endl;
    //   std::cout << " partie_positive_v " << partie_positive_v << std::endl;
    //   cas_PP = 1;
    //  }

    if (type == 0)  // flux arriere ou en dessous de cCells, integration entre
                    // -h0/2. et -h0/2.+abs(face_normal_velocity)*deltat_n
    {
      // Flux1m : integrale -inf,  -h0/2.+partie_positive_v
      flux1m =
          INT2Y(-h0 / 2. + partie_positive_v, -h0 / 2., phimoins[i], xg, yg);
      // Flux1m : integrale -inf,  -h0/2.
      flux1 = INT2Y(-h0 / 2., -h0 / 2., phimoins[i], xg, yg);
      //
      // if (abs(flux1-flux1m) > flux_threhold && abs(xg) != abs(xd)) {
      //   std::cout << i << " Activation Plateau Pente  AR Flux -h0/2, xg non
      //   nul " << cell << " = " << abs(flux1-flux1m) << std::endl;

      //   std::cout << " xg " << xg << " et h0/2 " << h0/2. << std::endl;
      //   std::cout << " xd " << xd << " et h0/2 " << h0/2. << std::endl;
      //   cas_PP = 1;
      //  }
      //
      // Flux2m : integrale -inf,  -h0/2.+partie_positive_v
      flux2m = INT2Y(-h0 / 2. + partie_positive_v, xg, yg, xd, yd);
      // Flux2 : integrale -inf,  -h0/2.
      flux2 = INT2Y(-h0 / 2., xg, yg, xd, yd);
      //
      // Flux3m : integrale -inf,  -h0/2.+partie_positive_v
      flux3m = INT2Y(-h0 / 2. + partie_positive_v, xd, yd, h0 / 2., phiplus[i]);
      // Flux3 : integrale -inf,  -h0/2.
      flux3 = INT2Y(-h0 / 2., xd, yd, h0 / 2., phiplus[i]);
      //
      // if (abs(flux3-flux3m) > flux_threhold && abs(xg) != abs(xd)) {
      //   std::cout << i << " Activation Plateau Pente  AR Flux xd , h0/2, non
      //   nul " << cell << " = " << abs(flux3-flux3m) << std::endl; std::cout
      //   << " xg " << xg << " et h0/2 " << h0/2. << std::endl; std::cout << "
      //   xd " << xd << " et h0/2 " << h0/2. << std::endl; cas_PP = 1;
      // }
      //
      // integrale positive
      Flux[i] = MathFunctions::max(
          ((flux1m - flux1) + (flux2m - flux2) + (flux3m - flux3)), 0.);
      // formule 16
      if (((phiplus[i] - phi[i]) * (phimoins[i] - phi[i])) >= 0.)
        Flux[i] = phi[i] * partie_positive_v;
      //
      // if (cas_PP == 1) std::cout << " type 1 Flux1 " << flux1 << " Flux1m "
      // << flux1m << " Flux2 " << flux2
      //			    << " Flux2m " << flux2m << " Flux3 " <<
      // flux3 << " Flux3m " << flux3m << " soit " << Flux[i] << std::endl;
    } else if (type ==
               1)  // flux devant ou au dessus de cCells, integration entre
                   // h0/2.-abs(face_normal_velocity)*deltat_n et h0/2.
    {
      // Flux1 : integrale -inf,  h0/2.-partie_positive_v
      flux1 = INT2Y(h0 / 2. - partie_positive_v, -h0 / 2., phimoins[i], xg, yg);
      // Flux1m : integrale -inf,  -h0/2.
      flux1m = INT2Y(h0 / 2., -h0 / 2., phimoins[i], xg, yg);
      //
      // if (abs(flux1-flux1m) > flux_threhold && abs(xg) != abs(xd)) {
      //   std::cout << i << " Activation Plateau Pente  AV Flux -h0/2, xg non
      //   nul " << cell << " = " << abs(flux1-flux1m) << std::endl; std::cout
      //   << " xg " << xg << " et h0/2 " << h0/2. << std::endl; std::cout << "
      //   xd " << xd << " et h0/2 " << h0/2. << std::endl; cas_PP = 1;
      //  }
      //
      // Flux2 : integrale -inf,  h0/2.-partie_positive_v
      flux2 = INT2Y(h0 / 2. - partie_positive_v, xg, yg, xd, yd);
      // Flux2m : integrale -inf,  -h0/2.
      flux2m = INT2Y(h0 / 2., xg, yg, xd, yd);
      //
      // Flux3 : integrale -inf,  h0/2.-partie_positive_v
      flux3 = INT2Y(h0 / 2. - partie_positive_v, xd, yd, h0 / 2., phiplus[i]);
      // Flux3m : integrale -inf,  -h0/2.
      flux3m = INT2Y(h0 / 2., xd, yd, h0 / 2., phiplus[i]);
      // if (abs(flux3-flux3m) > flux_threhold && abs(xg) != abs(xd)) {
      //    std::cout << i << " Activation Plateau Pente  AV Flux xd , h0/2, non
      //    nul " << cell << " = " << abs(flux3-flux3m) << std::endl;
      //  std::cout << " xg " << xg << " et h0/2 " << h0/2. << std::endl;
      //  std::cout << " xd " << xd << " et h0/2 " << h0/2. << std::endl;
      //  cas_PP = 1;
      // }
      //
      // integrale positive
      Flux[i] = MathFunctions::max(
          ((flux1m - flux1) + (flux2m - flux2) + (flux3m - flux3)), 0.);
      // formule 16
      if (((phiplus[i] - phi[i]) * (phimoins[i] - phi[i])) >= 0.)
        Flux[i] = phi[i] * partie_positive_v;

      // if (cas_PP == 1) std::cout << " type 1 Flux1 " << flux1 << " Flux1m "
      // << flux1m << " Flux2 " << flux2
      //			    << " Flux2m " << flux2m << " Flux3 " <<
      // flux3 << " Flux3m " << flux3m << " soit " << Flux[i] << std::endl;
    }

  }
  // std::cout << " Flux " << Flux << std::endl;
  // les flux de masse, de quantité de mouvement et d'energie massique se
  // deduisent des flux de volumes
  double somme_flux_masse = 0.;
  double somme_flux_volume = 0.;
  for (size_t imat = 0; imat < nbmat; imat++) {
    Flux[2 * nbmat + imat] =
        phi[2 * nbmat + imat] *
        Flux[nbmat + imat];  // flux de masse energy de imat
    somme_flux_masse += Flux[nbmat + imat];
    somme_flux_volume += Flux[imat];
  }
  Flux[3 * nbmat] =
      phi[3 * nbmat] * somme_flux_masse;  // flux de quantité de mouvement x
  Flux[3 * nbmat + 1] =
      phi[3 * nbmat + 1] * somme_flux_masse;  // flux de quantité de mouvement y
  Flux[3 * nbmat + 2] =
      phi[3 * nbmat + 2] * somme_flux_masse;  // flux d'energie cinetique
  Flux[3 * nbmat + 3] =
    phi[3 * nbmat + 3] * somme_flux_volume; // flux pour la pseudo VNR

  return Flux;
}

template <size_t d>
RealArray1D<d> Remap::computeUpwindFaceQuantities(
    RealArray1D<dim> face_normal, double face_normal_velocity, double delta_x,
    RealArray1D<dim> x_f, RealArray1D<d> phi_cb, RealArray1D<d> grad_phi_cb,
    RealArray1D<dim> x_cb, RealArray1D<d> phi_cf, RealArray1D<d> grad_phi_cf,
    RealArray1D<dim> x_cf) {
  if (face_normal_velocity * delta_x > 0.0)
    return phi_cb + (dot((x_f - x_cb), face_normal) * grad_phi_cb);
  else
    return phi_cf + (dot((x_f - x_cf), face_normal) * grad_phi_cf);
}

template <size_t d>
RealArray1D<d> Remap::computeRemapFlux(
    int projectionOrder, int projectionAvecPlateauPente,
    double face_normal_velocity, RealArray1D<dim> face_normal,
    double face_length, RealArray1D<d> phi_face,
    RealArray1D<dim> outer_face_normal, RealArray1D<dim> exy, double deltat_n) {
  RealArray1D<d> Flux;
  if (projectionAvecPlateauPente == 0) {
    // cas projection ordre 3 ou 1 ou 2 sans plateau pente (flux calculé ici
    // avec phi_face)
    if (MathFunctions::fabs(dot(face_normal, exy)) < 1.0E-10)
      return (0.0 * phi_face);
    return (dot(outer_face_normal, exy) * face_normal_velocity *
            face_length * deltat_n * phi_face);
  } else {
    // cas projection ordre 2 avec plateau pente (flux dans la variable
    // phi_face)
    if (MathFunctions::fabs(dot(face_normal, exy)) < 1.0E-10)
      return (0.0 * phi_face);
    return (dot(outer_face_normal, exy) * face_length * phi_face);
  }
}

#endif  // UTILESREMAP_IMPL_H
