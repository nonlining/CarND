#include "PID.h"

using namespace std;

/*
* TODO: Complete the PID class.
*/

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
	this->p_error = -(Kp * cte);
	this->i_error = -(Ki * sum);
	this->d_error = -(Kd*(cte - prev));
	
	this->prev = cte;
	
	//std::cout<<"update err "<<sum<<std::endl;
}

double PID::TotalError() {
	return (p_error + i_error + d_error);
}

