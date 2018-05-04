#include "PID.h"

using namespace std;


PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
  this->Kp = Kp;
  this->Ki = Ki;
  this->Kd = Kd;
  this->p_error = 0.0;
  this->i_error = 0.0;
  this->d_error = 0.0;
  //this->sum_squared_error = 0.0;
  
}

void PID::UpdateError(double cte) {

    this->p_error = this->Kp * cte;
    this->i_error += cte;
	this->d_error = this->Kd * (cte - this->p_error);
}

double PID::TotalError() {
    return this->p_error
           + this->Ki * this->i_error
           + this->d_error;
}

