#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1;
  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.
  
  //Complete the initialization. See ukf.h for other member properties.
  //Hint: one or more values initialized above might be wildly off...
  n_x_ = 5;
  
  n_aug_ = n_x_ + 2;
  
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  
  lambda_ = 3 - n_aug_;
  
  
  
  // Initialize weights
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
    for (int i = 1; i < 2*n_aug_+1; i++) {
	  weights_(i) = 0.5 / (n_aug_ + lambda_);
  }
  
  R_radar_ = MatrixXd(3, 3);
  R_radar_ << std_radr_*std_radr_, 0, 0,
              0, std_radphi_*std_radphi_, 0,
              0, 0,std_radrd_*std_radrd_;
  R_lidar_ = MatrixXd(2, 2);
  R_lidar_ << std_laspx_*std_laspx_,0,
              0,std_laspy_*std_laspy_;
}

UKF::~UKF() {}

void UKF::NormAng(double *ang) {
    while (*ang > M_PI) *ang -= 2. * M_PI;
    while (*ang < -M_PI) *ang += 2. * M_PI;
}
/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  //Complete this function! Make sure you switch between lidar and radar
  //measurements.
  
  if (!is_initialized_) {
	
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      
      //Convert radar from polar to cartesian coordinates and initialize state.
      double rho = meas_package.raw_measurements_[0];
      double theta = meas_package.raw_measurements_[1];
      double ro_dot = meas_package.raw_measurements_[2];
      x_(0) = rho * cos(theta);
      x_(1) = rho * sin(theta);
      x_(2) = sqrt(ro_dot * cos(theta) * ro_dot * cos(theta) + ro_dot * sin(theta) * ro_dot * sin(theta));
      x_(3) = 0;
	  x_(4) = 0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      //Initialize state.
	  double px = meas_package.raw_measurements_[0];
      double py = meas_package.raw_measurements_[1];
      x_ << px, py, 0, 0, 0;
	  if (fabs(x_(0)) < EPS){
		  x_(0) = EPS;
	  }
	  
	  if (fabs(x_(1)) < EPS){
		  x_(1) = EPS;
	  }
	  
    }

    previous_timestamp_ = meas_package.timestamp_;
    
    is_initialized_ = true;
	
	// covariance matrix initial 
	P_.fill(0.);
    P_(0,0) = 1.;
    P_(1,1) = 1.;
    P_(2,2) = 1.;
    P_(3,3) = 1.;
    P_(4,4) = 1.;
	
	
    return;
  }
  
  double dt = (meas_package.timestamp_ - previous_timestamp_) * 1e-6;
  previous_timestamp_ = meas_package.timestamp_;
  Prediction( dt );
  if( meas_package.sensor_type_ == MeasurementPackage::RADAR ) 
  {
    // Radar update
    UpdateRadar( meas_package );
  } 
  else 
  {
    // Laser update
    UpdateLidar( meas_package );
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  double delta_t2 = delta_t*delta_t;
  VectorXd x_aug = VectorXd(n_aug_);
  
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  
  x_aug.fill(0.0);
  x_aug.head(n_x_) = x_;
  P_aug.fill(0);
  P_aug.topLeftCorner(n_x_,n_x_) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;
  
  MatrixXd L = P_aug.llt().matrixL();
  
  Xsig_aug.col(0) = x_aug;
  double sqrt_lambda_n_aug = sqrt(lambda_+n_aug_);
  VectorXd sqrt_lambda_n_aug_L;
  
  for(int i = 0; i < n_aug_; i++) {
	sqrt_lambda_n_aug_L = sqrt_lambda_n_aug * L.col(i);
    Xsig_aug.col(i+1)        = x_aug + sqrt_lambda_n_aug_L;
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt_lambda_n_aug_L;
  }
  
  for (int i = 0; i< 2 * n_aug_ + 1; i++) {
    
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);
    
    double sin_yaw = sin(yaw);
    double cos_yaw = cos(yaw);
    double arg = yaw + yawd*delta_t;
    
    
    double px_p, py_p;
    
    if (fabs(yawd) > EPS) {	
	double v_yawd = v/yawd;
        px_p = p_x + v_yawd * (sin(arg) - sin_yaw);
        py_p = p_y + v_yawd * (cos_yaw - cos(arg) );
    } else {
	double v_delta_t = v*delta_t;
        px_p = p_x + v_delta_t*cos_yaw;
        py_p = p_y + v_delta_t*sin_yaw;
    }
    double v_p = v;
    double yaw_p = arg;
    double yawd_p = yawd;

    px_p += 0.5*nu_a*delta_t2*cos_yaw;
    py_p += 0.5*nu_a*delta_t2*sin_yaw;
    v_p += nu_a*delta_t;
    yaw_p += 0.5*nu_yawdd*delta_t2;
    yawd_p += nu_yawdd*delta_t;

    
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }
  
  
  x_ = Xsig_pred_ * weights_;
  
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    
    NormAng(&(x_diff(3)));
    P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
}

void UKF::Update(MeasurementPackage meas_package, MatrixXd Zsig, int n_z){

  VectorXd z_pred = VectorXd(n_z);
  z_pred  = Zsig * weights_;
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) { 
    VectorXd z_diff = Zsig.col(i) - z_pred;
    NormAng(&(z_diff(1)));
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  MatrixXd R = MatrixXd(n_z, n_z);
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR){ // Radar
    R = R_radar_;
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::LASER){ // Lidar
    R = R_lidar_;
  }
  S = S + R;
  

  MatrixXd Tc = MatrixXd(n_x_, n_z);

  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) { 

    VectorXd z_diff = Zsig.col(i) - z_pred;
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR){ // Radar

      NormAng(&(z_diff(1)));
    }

    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    NormAng(&(x_diff(3)));
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  VectorXd z = meas_package.raw_measurements_;

  MatrixXd K = Tc * S.inverse();

  VectorXd z_diff = z - z_pred;
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR){ // Radar

    NormAng(&(z_diff(1)));
  }

  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR){ // Radar
	NIS_radar_ = z.transpose() * S.inverse() * z;
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::LASER){ // Lidar
	NIS_laser_ = z.transpose() * S.inverse() * z;
  }
}