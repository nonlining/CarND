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
  this->sum = 0;
  this->prev = 0;

}

void PID::UpdateError(double cte) {
	
	sum += cte;

    this->p_error = - Kp * cte;
    this->i_error = - Ki * sum;
    this->d_error = - Kd * (cte - prev);
	this->prev = cte;
}

double PID::TotalError() {
    return this->p_error + this->i_error + this->d_error;
}

