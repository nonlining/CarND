#ifndef MPC_H
#define MPC_H

#include <vector>
#include "Eigen-3.3/Eigen/Core"

#define DT 0.1
#define Lf 2.67
#define NUMBER_OF_STEPS 20
#define REF_CTE 0
#define REF_EPSI 0
#define REF_V 77.5
#define W_CTE 8.4
#define W_EPSI 0.32
#define W_V 0.261
#define W_DELTA 600000
#define W_A 17.1
#define W_DDELTA 0.01
#define W_DA 0.00001
#define DEBUG 0

using namespace std;

class MPC {
 public:
  MPC();

  virtual ~MPC();

  // Solve the model given an initial state and polynomial coefficients.
  // Return the first actuatotions.
  vector<double> Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs);
  vector<double> mpc_x;
  vector<double> mpc_y;
};

#endif /* MPC_H */
