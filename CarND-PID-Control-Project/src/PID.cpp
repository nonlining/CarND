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
  this->sum_squared_error = 0.0;
  this->n = 1;
}

void PID::UpdateError(double cte) {

    sum_squared_error += cte * cte;
    //double avg_error = sum_squared_error / n;

    n++;

    this->d_error = cte - this->p_error;
    this->p_error = cte;
    this->i_error += cte;
}

double PID::TotalError() {
    return this->Kp * this->p_error
           + this->Ki * this->i_error
           + this->Kd * this->d_error;
}

