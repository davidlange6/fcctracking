"""Single-particle generator writing HEPEvt (long format, .hepevt) for ddsim.

One particle per event:
  momentum  fixed (--p) or a mixture over [pmin, pmax]: a fraction --log-frac
            log-uniform, the rest uniform (sampling in p, not pT)
  direction uniform in cos(theta) with |cos(theta)| <= --cos-max, phi uniform
  charge    PDG sign chosen at random per event for --pdg-abs (e.g. 13 -> mu+/mu-),
            or fixed with --pdg
  vertex    Gaussian, sigma_xy / sigma_z in mm, t = 0

Long-format layout read by DD4hep's Geant4EventReaderHepEvtLong, per event:
  NHEP
  ISTHEP IDHEP JMOHEP1 JMOHEP2 JDAHEP1 JDAHEP2 PX PY PZ E M VX VY VZ T   (GeV, mm)
"""

import argparse

import numpy as np

MASSES = {11: 0.000510999, 13: 0.1056584, 211: 0.13957039, 321: 0.493677, 2212: 0.93827209}


def sample(n, rng, p_fixed, pmin, pmax, log_frac, cos_max, sigma_xy, sigma_z):
    if p_fixed is not None:
        p = np.full(n, p_fixed)
    else:
        log = rng.random(n) < log_frac
        p = np.where(log, np.exp(rng.uniform(np.log(pmin), np.log(pmax), n)), rng.uniform(pmin, pmax, n))
    cos_t = rng.uniform(-cos_max, cos_max, n)
    sin_t = np.sqrt(1 - cos_t**2)
    phi = rng.uniform(0, 2 * np.pi, n)
    mom = p[:, None] * np.stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], 1)
    vtx = rng.normal(0.0, 1.0, (n, 3)) * np.array([sigma_xy, sigma_xy, sigma_z])
    return mom, vtx


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    charge = ap.add_mutually_exclusive_group(required=True)
    charge.add_argument("--pdg", type=int, help="fixed PDG id")
    charge.add_argument("--pdg-abs", type=int, help="|PDG id|; sign chosen at random per event")
    ap.add_argument("--p", type=float, help="fixed momentum [GeV]")
    ap.add_argument("--pmin", type=float, default=0.2)
    ap.add_argument("--pmax", type=float, default=50.0)
    ap.add_argument("--log-frac", type=float, default=0.5)
    ap.add_argument("--cos-max", type=float, default=0.99)
    ap.add_argument("--sigma-xy", type=float, default=1.0, help="vertex sigma in x and y [mm]")
    ap.add_argument("--sigma-z", type=float, default=20.0, help="vertex sigma in z [mm]")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    mom, vtx = sample(args.n, rng, args.p, args.pmin, args.pmax, args.log_frac, args.cos_max,
                      args.sigma_xy, args.sigma_z)  # fmt: skip
    if args.pdg is not None:
        pdg = np.full(args.n, args.pdg)
    else:
        pdg = np.where(rng.random(args.n) < 0.5, args.pdg_abs, -args.pdg_abs)
    with open(args.out, "w") as f:
        for i in range(args.n):
            m = MASSES[abs(int(pdg[i]))]
            px, py, pz = mom[i]
            e = np.sqrt(px * px + py * py + pz * pz + m * m)
            vx, vy, vz = vtx[i]
            f.write("1\n")
            f.write(f"1 {int(pdg[i])} 0 0 0 0 {px:.12e} {py:.12e} {pz:.12e} {e:.12e} {m:.9e} "
                    f"{vx:.12e} {vy:.12e} {vz:.12e} 0.0\n")  # fmt: skip


if __name__ == "__main__":
    main()
