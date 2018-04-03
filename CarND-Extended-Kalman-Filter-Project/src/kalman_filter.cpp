#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
    // predict the state
	x_ = F_ * x_;
    MatrixXd Ft = F_.transpose();
	P_ = F_ * P_ * Ft + Q_;

}

void KalmanFilter::Update(const VectorXd &z) {
    //update the state by using Kalman Filter equations
    VectorXd y = z - H_ * x_;
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd Si = S.inverse();
    MatrixXd PHt = P_ * Ht;
    MatrixXd K = PHt * Si;
	
    x_ = x_ + (K * y);
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;

}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
    //update the state by using Extended Kalman Filter equations
    double p_x = x_[0];
    double p_y = x_[1];
    double v_x = x_[2];
    double v_y = x_[3];

    if (fabs(p_x) < 0.0001) {
        p_x = 0.0001;
    }

    double rho = sqrt(p_x * p_x + p_y * p_y);
    
	if (fabs(rho) < 0.0001) {
        rho = 0.0001;
    }

    double phi = atan(p_y / p_x);
    double rho_dot = (p_x * v_x + p_y * v_y) / rho;

    VectorXd h(3);
    h << rho, phi, rho_dot;

    VectorXd y = z - h;
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd Si = S.inverse();
    MatrixXd PHt = P_ * Ht;
    MatrixXd K = PHt * Si;

    x_ = x_ + (K * y);
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;
  
}
